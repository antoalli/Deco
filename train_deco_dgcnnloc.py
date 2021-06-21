"""
@Revision Author: Antonio Alliegro
@File:
@Time:
"""
import open3d as o3  # mandatory to import (open3d 0.9.0) before torch (1.2) to avoid crash!
import argparse
import json
import shutil
import os
import random
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch_nndistance as NND
from tensorboardX import SummaryWriter
import shapenet_part_loader
from shape_utils import random_occlude_pointcloud as crop_shape
from utils import IOStream, safe_make_dirs
from model_utils import remove_prefix_dict
from models.model_deco import GLEncoder_dgcnn as Encoder, Generator
silent_warn = o3


"""
train DeCo with DGCNN at local encoder - ablation experiment
For 'Local Denoising Variants' (table 8 arxiv) experiments in which we compared different GCN networks at local encoder
"""


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manualSeed', type=int, help='manual seed')
    parser.add_argument('--batch_size', type=int, default=30)
    parser.add_argument('--epochs', type=int, default=241)
    parser.add_argument('--workers', type=int, default=6, help='num of workers to load data for each DataLoader')
    parser.add_argument('--checkpoints_dir', '-CDIR', default='experiments_deco', help='Folder where all experiments get stored')
    parser.add_argument('--exp_name', '-EXP', default='exp', help='will create an exp_name folder under checkpoints_dir')
    parser.add_argument('--config', '-C', required=True, help='path to valid configuration file')
    parser.add_argument('--parallel', action='store_true', help="Multi-GPU Training")
    parser.add_argument('--it_test', type=int, default=10, help='at each it_test epoch: perform test and checkpoint')
    parser.add_argument('--restart_from', default='', help='restart interrupted training from checkpoint')
    parser.add_argument('--class_choice',
                        default="Airplane,Bag,Cap,Car,Chair,Guitar,Lamp,Laptop,Motorbike,Mug,Pistol,Skateboard,Table",
                        help='Classes to train on: default is 13 classes used in PF-Net')
    parser.add_argument('--data_root', default="/home/antonioa/data/shapenetcore_partanno_segmentation_benchmark_v0")

    # crop params
    parser.add_argument('--crop_point_num', type=int, default=512, help='number of points to crop')
    parser.add_argument('--context_point_num', type=int, default=512, help='number of points of the frame region')
    parser.add_argument('--num_holes', type=int, default=1, help='number of crop_point_num holes')
    parser.add_argument('--pool1_points', '-P1', type=int, default=1280,
                        help='points selected at pooling layer 1, we use 1280 in all experiments')
    parser.add_argument('--pool2_points', '-P2', type=int, default=512,
                        help='points selected at pooling layer 2, should match crop_point_num i.e. 512')
    # parser.add_argument('--fps_centroids', '-FPS', action='store_true', help='different crop logic than pfnet')
    parser.add_argument('--raw_weight', '-RW', type=float, default=1,
                        help='weights the intermediate pred (frame reg.) loss, use 0 this to disable regularization.')

    args = parser.parse_args()
    args.fps_centroids = False

    # make experiment dirs
    args.save_dir = os.path.join(args.checkpoints_dir, args.exp_name)
    args.models_dir = os.path.join(args.save_dir, 'models')
    args.vis_dir = os.path.join(args.save_dir, 'train_visz')
    safe_make_dirs([args.save_dir, args.models_dir, args.vis_dir, os.path.join(args.save_dir, 'backup_code')])

    # instantiate loggers
    io_logger = IOStream(os.path.join(args.save_dir, 'log.txt'))
    tb_logger = SummaryWriter(logdir=args.save_dir)

    return args, io_logger, tb_logger


def weights_init_normal(m):
    """ Weights initialization with normal distribution.. Xavier """
    classname = m.__class__.__name__
    if classname.find("Conv2d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("Conv1d") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
    elif classname.find("BatchNorm1d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


def main_worker():
    opt, io, tb = get_args()
    start_epoch = -1
    start_time = time.time()
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # python script folder
    ckt = None
    if len(opt.restart_from) > 0:
        ckt = torch.load(opt.restart_from)
        start_epoch = ckt['epoch'] - 1

    # load configuration from file
    try:
        with open(opt.config) as cf:
            config = json.load(cf)
    except IOError as error:
        print(error)

    # backup relevant files
    shutil.copy(src=os.path.abspath(__file__), dst=os.path.join(opt.save_dir, 'backup_code'))
    shutil.copy(src=os.path.join(BASE_DIR, 'models', 'model_deco.py'),
                dst=os.path.join(opt.save_dir, 'backup_code'))
    shutil.copy(src=os.path.join(BASE_DIR, 'shape_utils.py'),
                dst=os.path.join(opt.save_dir, 'backup_code'))
    shutil.copy(src=opt.config, dst=os.path.join(opt.save_dir, 'backup_code', 'config.json.backup'))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)
    torch.cuda.manual_seed_all(opt.manualSeed)

    io.cprint(f"Arguments: {str(opt)}")
    io.cprint(f"Configuration: {str(config)}")

    pnum = config['completion_trainer']['num_points']
    class_choice = opt.class_choice
    # datasets + loaders
    if len(class_choice) > 0:
        class_choice = ''.join(opt.class_choice.split()).split(",")  # sanitize + split(",")
        io.cprint("Class choice list: {}".format(str(class_choice)))
    else:
        class_choice = None  # Train on all classes! (if opt.class_choice=='')

    tr_dataset = shapenet_part_loader.PartDataset(root=opt.data_root,
                                                  classification=True,
                                                  class_choice=class_choice,
                                                  npoints=pnum,
                                                  split='train')

    te_dataset = shapenet_part_loader.PartDataset(root=opt.data_root,
                                                  classification=True,
                                                  class_choice=class_choice,
                                                  npoints=pnum,
                                                  split='test')

    tr_loader = torch.utils.data.DataLoader(tr_dataset,
                                            batch_size=opt.batch_size,
                                            shuffle=True,
                                            num_workers=opt.workers,
                                            drop_last=True)

    te_loader = torch.utils.data.DataLoader(te_dataset,
                                            batch_size=64,
                                            shuffle=True,
                                            num_workers=opt.workers)

    num_holes = int(opt.num_holes)
    crop_point_num = int(opt.crop_point_num)
    context_point_num = int(opt.context_point_num)
    # io.cprint("Num holes: {}".format(num_holes))
    # io.cprint("Crop points num: {}".format(crop_point_num))
    # io.cprint("Context points num: {}".format(context_point_num))
    # io.cprint("Pool1 num points selected: {}".format(opt.pool1_points))
    # io.cprint("Pool2 num points selected: {}".format(opt.pool2_points))

    """" Models """
    gl_encoder = Encoder(conf=config)
    generator = Generator(conf=config, pool1_points=int(opt.pool1_points), pool2_points=int(opt.pool2_points))
    gl_encoder.apply(weights_init_normal)  # affecting only non pretrained
    generator.apply(weights_init_normal)  # not pretrained
    print("Encoder: ", gl_encoder)
    print("Generator: ", generator)

    if ckt is not None:
        io.cprint(f"Restart Training from epoch {start_epoch}.")
        gl_encoder.load_state_dict(ckt['gl_encoder_state_dict'])
        generator.load_state_dict(ckt['generator_state_dict'])
    else:
        io.cprint("Training Completion Task...")
        local_fe_fn = config['completion_trainer']['checkpoint_local_enco']
        global_fe_fn = config['completion_trainer']['checkpoint_global_enco']

        if len(local_fe_fn) > 0:
            local_enco_dict = torch.load(local_fe_fn)['model_state_dict']
            # refactoring pretext-trained local dgcnn encoder state dict keys
            local_enco_dict = remove_prefix_dict(state_dict=local_enco_dict, to_remove_str='local_encoder.')
            loc_load_result = gl_encoder.local_encoder.load_state_dict(local_enco_dict, strict=False)
            io.cprint(f"Local FE pretrained weights - loading res: {str(loc_load_result)}")
        else:
            # Ablation experiments only
            io.cprint("Local FE pretrained weights - NOT loaded", color='r')

        if len(global_fe_fn) > 0:
            global_enco_dict = torch.load(global_fe_fn, )['global_encoder']
            glob_load_result = gl_encoder.global_encoder.load_state_dict(global_enco_dict, strict=True)
            io.cprint(f"Global FE pretrained weights - loading res: {str(glob_load_result)}", color='b')
        else:
            # Ablation experiments only
            io.cprint("Global FE pretrained weights - NOT loaded", color='r')

    io.cprint("Num GPUs: " + str(torch.cuda.device_count()) + ", Parallelism: {}".format(opt.parallel))
    if opt.parallel and torch.cuda.device_count() > 1:
        gl_encoder = torch.nn.DataParallel(gl_encoder)
        generator = torch.nn.DataParallel(generator)
    gl_encoder.to(device)
    generator.to(device)

    # Optimizers + schedulers
    opt_E = torch.optim.Adam(
        gl_encoder.parameters(),
        lr=config['completion_trainer']['enco_lr'],  # def: 10e-4
        betas=(0.9, 0.999),
        eps=1e-05,
        weight_decay=0.001)

    sched_E = torch.optim.lr_scheduler.StepLR(
        opt_E,
        step_size=config['completion_trainer']['enco_step'],  # def: 25
        gamma=0.5)

    opt_G = torch.optim.Adam(
        generator.parameters(),
        lr=config['completion_trainer']['gen_lr'],  # def: 10e-4
        betas=(0.9, 0.999),
        eps=1e-05,
        weight_decay=0.001)

    sched_G = torch.optim.lr_scheduler.StepLR(
        opt_G,
        step_size=config['completion_trainer']['gen_step'],  # def: 40
        gamma=0.5)

    if ckt is not None:
        opt_E.load_state_dict(ckt['optimizerE_state_dict'])
        opt_G.load_state_dict(ckt['optimizerG_state_dict'])
        sched_E.load_state_dict(ckt['schedulerE_state_dict'])
        sched_G.load_state_dict(ckt['schedulerG_state_dict'])

    if not opt.fps_centroids:
        # 5 viewpoints to crop around - same as in PFNet
        centroids = np.asarray(
            [[1, 0, 0], [0, 0, 1], [1, 0, 1], [-1, 0, 0], [-1, 1, 0]])
    else:
        raise NotImplementedError('experimental')
        centroids = None

    io.cprint("Training.. \n")
    best_test = sys.float_info.max
    best_ep = -1
    it = 0  # global iteration counter
    vis_folder = None
    for epoch in range(start_epoch + 1, opt.epochs):
        start_ep_time = time.time()
        count = 0.0
        tot_loss = 0.0
        tot_fine_loss = 0.0
        tot_raw_loss = 0.0
        gl_encoder = gl_encoder.train()
        generator = generator.train()
        for i, data in enumerate(tr_loader, 0):
            it += 1
            points, _ = data
            B, N, dim = points.size()
            count += B

            partials = []
            fine_gts, raw_gts = [], []
            N_partial_points = N - (crop_point_num * num_holes)
            for m in range(B):
                # points[m]: complete shape of size (N,3)
                # partial: partial point cloud to complete
                # fine_gt: missing part ground truth
                # raw_gt: missing part ground truth + frame points (where frame points are points included in partial)
                partial, fine_gt, raw_gt = crop_shape(
                    points[m],
                    centroids=centroids,
                    scales=[crop_point_num, (crop_point_num + context_point_num)],
                    n_c=num_holes)

                if partial.size(0) > N_partial_points:
                    assert num_holes > 1, "Should be no need to resample if not multiple holes case"
                    # sampling without replacement
                    choice = torch.randperm(partial.size(0))[:N_partial_points]
                    partial = partial[choice]

                partials.append(partial)
                fine_gts.append(fine_gt)
                raw_gts.append(raw_gt)

            if i == 1 and epoch % opt.it_test == 0:
                # make some visualization
                vis_folder = os.path.join(opt.vis_dir, "epoch_{}".format(epoch))
                safe_make_dirs([vis_folder])
                print(f"ep {epoch} - Saving visualizations into: {vis_folder}")
                for j in range(len(partials)):
                    np.savetxt(
                        X=partials[j], fname=os.path.join(vis_folder, '{}_cropped.txt'.format(j)),
                        fmt='%.5f', delimiter=';')
                    np.savetxt(
                        X=fine_gts[j], fname=os.path.join(vis_folder, '{}_fine_gt.txt'.format(j)),
                        fmt='%.5f', delimiter=';')
                    np.savetxt(
                        X=raw_gts[j], fname=os.path.join(vis_folder, '{}_raw_gt.txt'.format(j)),
                        fmt='%.5f', delimiter=';')

            partials = torch.stack(partials).to(device).permute(0, 2, 1)  # [B, 3, N-512]
            fine_gts = torch.stack(fine_gts).to(device)  # [B, 512, 3]
            raw_gts = torch.stack(raw_gts).to(device)  # [B, 512 + context, 3]

            if i == 1:  # sanity check
                print("[dbg]: partials: ", partials.size(), ' ', partials.device)
                print("[dbg]: fine grained gts: ", fine_gts.size(), ' ', fine_gts.device)
                print("[dbg]: raw grained gts: ", raw_gts.size(), ' ', raw_gts.device)

            gl_encoder.zero_grad()
            generator.zero_grad()
            feat = gl_encoder(partials)
            fake_fine, fake_raw = generator(feat)  # pred_fine (only missing part), pred_intermediate (missing + frame)

            # pytorch 1.2 compiled Chamfer (C2C) dist.
            assert fake_fine.size() == fine_gts.size(), "Wrong input shapes to Chamfer module"
            if i == 0:
                if fake_raw.size() != raw_gts.size():
                    warnings.warn("size dismatch for: raw_pred: {}, raw_gt: {}".format(
                        str(fake_raw.size()), str(raw_gts.size())))

            # fine grained prediction + gt
            fake_fine = fake_fine.contiguous()
            fine_gts = fine_gts.contiguous()
            # raw prediction + gt
            fake_raw = fake_raw.contiguous()
            raw_gts = raw_gts.contiguous()

            dist1, dist2, _, _ = NND.nnd(fake_fine, fine_gts)  # fine grained loss computation
            dist1_raw, dist2_raw, _, _ = NND.nnd(fake_raw, raw_gts)  # raw grained loss computation

            # standard C2C distance loss
            fine_loss = 100 * (0.5 * torch.mean(dist1) + 0.5 * torch.mean(dist2))

            # raw loss: missing part + frame
            raw_loss = 100 * (0.5 * torch.mean(dist1_raw) + 0.5 * torch.mean(dist2_raw))

            loss = fine_loss + opt.raw_weight * raw_loss  # missing part pred loss + α * raw reconstruction loss
            loss.backward()
            opt_E.step()
            opt_G.step()
            tot_loss += loss.item() * B
            tot_fine_loss += fine_loss.item() * B
            tot_raw_loss += raw_loss.item() * B

            if it % 10 == 0:
                io.cprint('[%d/%d][%d/%d]: loss: %.4f, fine CD: %.4f, interm. CD: %.4f' % (
                    epoch, opt.epochs, i, len(tr_loader),
                    loss.item(),
                    fine_loss.item(),
                    raw_loss.item()))

            # make visualizations
            if i == 1 and epoch % opt.it_test == 0:
                assert (vis_folder is not None and os.path.exists(vis_folder))
                fake_fine = fake_fine.cpu().detach().data.numpy()
                fake_raw = fake_raw.cpu().detach().data.numpy()
                for j in range(len(fake_fine)):
                    np.savetxt(
                        X=fake_fine[j], fname=os.path.join(vis_folder, '{}_pred_fine.txt'.format(j)), fmt='%.5f',
                        delimiter=';')
                    np.savetxt(
                        X=fake_raw[j], fname=os.path.join(vis_folder, '{}_pred_raw.txt'.format(j)), fmt='%.5f',
                        delimiter=';')

        sched_E.step()
        sched_G.step()
        io.cprint('[%d/%d] Ep Train - loss: %.5f, fine cd: %.5f, interm. cd: %.5f' %
                  (epoch, opt.epochs, tot_loss * 1.0 / count, tot_fine_loss * 1.0 / count, tot_raw_loss * 1.0 / count))
        tb.add_scalar('Train/tot_loss', tot_loss * 1.0 / count, epoch)
        tb.add_scalar('Train/cd_fine', tot_fine_loss * 1.0 / count, epoch)
        tb.add_scalar('Train/cd_interm', tot_raw_loss * 1.0 / count, epoch)

        if epoch % opt.it_test == 0:
            torch.save(
                {
                    'type_exp': 'dgccn at local encoder',
                    'epoch': epoch + 1,
                    'epoch_train_loss': tot_loss * 1.0 / count,
                    'epoch_train_loss_raw': tot_raw_loss * 1.0 / count,
                    'epoch_train_loss_fine': tot_fine_loss * 1.0 / count,
                    'gl_encoder_state_dict': gl_encoder.module.state_dict() if isinstance(gl_encoder, nn.DataParallel)
                    else gl_encoder.state_dict(),
                    'generator_state_dict': generator.module.state_dict() if isinstance(generator, nn.DataParallel)
                    else generator.state_dict(),
                    'optimizerE_state_dict': opt_E.state_dict(),
                    'optimizerG_state_dict': opt_G.state_dict(),
                    'schedulerE_state_dict': sched_E.state_dict(),
                    'schedulerG_state_dict': sched_G.state_dict(),
                }, os.path.join(opt.models_dir, 'checkpoint_' + str(epoch) + '.pth'))

        if epoch % opt.it_test == 0:
            test_cd, count = 0.0, 0.0
            for i, data in enumerate(te_loader, 0):
                points, _ = data
                B, N, dim = points.size()
                count += B

                partials = []
                fine_gts = []
                N_partial_points = N - (crop_point_num * num_holes)

                for m in range(B):
                    partial, fine_gt, _ = crop_shape(
                        points[m],
                        centroids=centroids,
                        scales=[crop_point_num, (crop_point_num + context_point_num)],
                        n_c=num_holes)

                    if partial.size(0) > N_partial_points:
                        assert num_holes > 1
                        # sampling Without replacement
                        choice = torch.randperm(partial.size(0))[:N_partial_points]
                        partial = partial[choice]

                    partials.append(partial)
                    fine_gts.append(fine_gt)
                partials = torch.stack(partials).to(device).permute(0, 2, 1)  # [B, 3, N-512]
                fine_gts = torch.stack(fine_gts).to(device).contiguous()  # [B, 512, 3]

                # TEST FORWARD
                # Considering only missing part prediction at Test Time
                gl_encoder.eval()
                generator.eval()
                with torch.no_grad():
                    feat = gl_encoder(partials)
                    fake_fine, _ = generator(feat)

                fake_fine = fake_fine.contiguous()
                assert fake_fine.size() == fine_gts.size()
                dist1, dist2, _, _ = NND.nnd(fake_fine, fine_gts)
                cd_loss = 100 * (0.5 * torch.mean(dist1) + 0.5 * torch.mean(dist2))
                test_cd += cd_loss.item() * B

            test_cd = test_cd * 1.0 / count
            io.cprint('Ep Test [%d/%d] - cd loss: %.5f ' % (epoch, opt.epochs, test_cd), color="b")
            tb.add_scalar('Test/cd_loss', test_cd, epoch)
            is_best = test_cd < best_test
            best_test = min(best_test, test_cd)

            if is_best:
                # best model case
                best_ep = epoch
                io.cprint("New best test %.5f at epoch %d" % (best_test, best_ep))
                shutil.copyfile(
                    src=os.path.join(opt.models_dir, 'checkpoint_' + str(epoch) + '.pth'),
                    dst=os.path.join(opt.models_dir, 'best_model.pth'))
        io.cprint('[%d/%d] Epoch time: %s' % (
            epoch, num_epochs, time.strftime("%M:%S", time.gmtime(time.time() - start_ep_time))))

    # Script ends
    hours, rem = divmod(
        time.time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    io.cprint("### Training ended in {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    io.cprint("### Best val %.6f at epoch %d" % (best_test, best_ep))


if __name__ == '__main__':
    # train DeCo with DGCNN at local encoder
    # ablation experiment: comparing diff. graph conv. at local enco
    main_worker()
