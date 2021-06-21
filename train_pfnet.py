import open3d as o3
import os
import time
import argparse
import random
import torch
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import utils
import shapenet_part_loader
from models.model_pfnet import _netlocalD, _netG
from tensorboardX import SummaryWriter
from shape_utils import random_occlude_pointcloud_v2 as crop_shape
from utils import IOStream
import shutil
import torch_nndistance as NND
import math
import numpy as np
silent_warn = o3


parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=16, help='number of data loading workers')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--pnum', type=int, default=2048, help='the point number of a sample')
parser.add_argument('--niter', type=int, default=201, help='number of epochs to train for')
parser.add_argument('--weight_decay', type=float, default=0.001)
parser.add_argument('--learning_rate', default=0.0002, type=float, help='learning rate in training')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--D_choose', type=int, default=1, help='0 not use D-net,1 use D-net')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--drop', type=float, default=0.2)
parser.add_argument('--num_scales', type=int, default=3, help='number of scales')
parser.add_argument('--point_scales_list', type=list, default=[2048, 1024, 512],
                    help='number of points in each scales')
parser.add_argument('--each_scales_size', type=int, default=1, help='each scales size')
parser.add_argument('--wtl2', type=float, default=0.95, help='0 means do not use else use with this weight')
# parser.add_argument('--cropmethod', default='random_center', help='random|center|random_center')
parser.add_argument('--checkpoints_dir', default='checkpoints', help='checkpoints dir')
parser.add_argument('--exp_name', default='exp', help='experiment name')
parser.add_argument('--data_root',
                    default='/home/alliegro/data/shapenetcore_partanno_segmentation_benchmark_v0/',
                    help='dataset path')
parser.add_argument('--class_choice',
                    default="Airplane,Bag,Cap,Car,Chair,Guitar,Lamp,Laptop,Motorbike,Mug,Pistol,Skateboard,Table",
                    help='Classes to train on: default is 13 classes used in PF-Net')
# crop params
parser.add_argument('--crop_point_num', type=int, default=512, help='0 means do not use else use with this weight')
parser.add_argument('--num_holes', type=int, default=1)
parser.add_argument('--fps_centers', '-FPS', action='store_true', help="farthest points as crop centroids")
opt = parser.parse_args()

# make experiment dirs
save_dir = os.path.join(opt.checkpoints_dir, opt.exp_name)
point_netG_saving = os.path.join(save_dir, 'models', 'point_netG')
point_netD_saving = os.path.join(save_dir, 'models', 'point_netD')
visz_folder = None

print('Experiment results and checkpoints stored in {}'.format(save_dir))
if not os.path.exists(os.path.join(save_dir, 'models')):
    os.makedirs(os.path.join(save_dir, 'models'))
if not os.path.exists(point_netG_saving):
    os.makedirs(point_netG_saving)
if not os.path.exists(point_netD_saving):
    os.makedirs(point_netD_saving)
if not os.path.exists(os.path.join(save_dir, 'backup-code')):
    os.makedirs(os.path.join(save_dir, 'backup-code'))
if not os.path.exists(os.path.join(save_dir, "train_visz")):
    os.makedirs(os.path.join(save_dir, "train_visz"))

filename = os.path.abspath(__file__).split('/')[-1]
os.system(
    'cp {} {}'.format(os.path.abspath(__file__), os.path.join(save_dir, 'backup-code', '{}.backup'.format(filename))))

io = IOStream(os.path.join(save_dir, 'log.txt'))
tb = SummaryWriter(logdir=save_dir)
io.cprint("PFNet training -\n num holes: %d, cropped points around each: %d" % (opt.num_holes, opt.crop_point_num))
io.cprint('-' * 30)
io.cprint('Arguments: ')
io.cprint(str(opt) + '\n')

USE_CUDA = True
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
point_netG = _netG(opt.num_scales, opt.each_scales_size, opt.point_scales_list, opt.crop_point_num * opt.num_holes)
if opt.D_choose == 1:
    point_netD = _netlocalD(opt.crop_point_num * opt.num_holes)
resume_epoch = 0


def weights_init_normal(m):
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


if USE_CUDA:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    if opt.ngpu > 1:
        io.cprint('Using DataParallel')
        io.cprint('Num GPUs: %d' % torch.cuda.device_count())
        point_netG = torch.nn.DataParallel(point_netG)
        if opt.D_choose == 1: point_netD = torch.nn.DataParallel(point_netD)
    point_netG.to(device)
    point_netG.apply(weights_init_normal)
    if opt.D_choose == 1:
        point_netD.to(device)
        point_netD.apply(weights_init_normal)

if opt.netG != '':
    point_netG.load_state_dict(torch.load(opt.netG, map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netG)['epoch']

if opt.netD != '':
    point_netD.load_state_dict(torch.load(opt.netD, map_location=lambda storage, location: storage)['state_dict'])
    resume_epoch = torch.load(opt.netD)['epoch']

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)

io.cprint("Random Seed: %d" % opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

""" Datasets and Loader """
if len(opt.class_choice) > 0:
    class_choice = ''.join(
        opt.class_choice.split()).split(",")  # sanitize + split(",")
    io.cprint("Class choice: {}\n".format(str(class_choice)))
else:
    class_choice = None  # iff. opt.class_choice=='' train on all classes

dset = shapenet_part_loader.PartDataset(
    root=opt.data_root,
    classification=True,
    class_choice=class_choice,
    npoints=opt.pnum,
    split='train')

train_loader = torch.utils.data.DataLoader(
    dset,
    batch_size=opt.batchSize,
    shuffle=True,
    num_workers=int(opt.workers),
    drop_last=True,
)

test_dset = shapenet_part_loader.PartDataset(
    root=opt.data_root,
    classification=True,
    class_choice=class_choice,
    npoints=opt.pnum,
    split='test')

test_dataloader = torch.utils.data.DataLoader(
    test_dset,
    batch_size=int(opt.batchSize) * 2,
    shuffle=False,
    num_workers=int(opt.workers))

io.cprint("\nGenerator:")
io.cprint(str(point_netG))

criterion = torch.nn.BCEWithLogitsLoss().to(device)  # used at Discriminator

# setup optimizer + schedulers
optimizerG = torch.optim.Adam(
    point_netG.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-05, weight_decay=opt.weight_decay
)
schedulerG = torch.optim.lr_scheduler.StepLR(optimizerG, step_size=40, gamma=0.2)

if opt.D_choose == 1:
    optimizerD = torch.optim.Adam(
        point_netD.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-05, weight_decay=opt.weight_decay
    )
    schedulerD = torch.optim.lr_scheduler.StepLR(optimizerD, step_size=40, gamma=0.2)
    io.cprint("\nDiscriminator:")
    io.cprint(str(point_netD) + '\n')

real_label = 1
fake_label = 0
crop_point_num = int(opt.crop_point_num)
num_holes = int(opt.num_holes)
label = torch.FloatTensor(opt.batchSize)
num_batch = len(dset) / opt.batchSize
it = 0  # iteration counter
visz_folder = None
num_it = math.floor(len(train_loader.dataset) / opt.batchSize) \
    if train_loader.drop_last else math.ceil(len(train_loader.dataset) / opt.batchSize)

# Viewpoints to crop around
# same as in PFNet
if not opt.fps_centers:
    centroids = np.asarray(
        [[1, 0, 0], [0, 0, 1], [1, 0, 1], [-1, 0, 0], [-1, 1, 0]])
else:
    raise NotImplementedError('experimental')
    io.cprint("Shape itself 10-FPS as centroids for crop", color="r")
    centroids = None

# Overall train stats
best_test = 999999.0
best_ep = -1

if opt.D_choose == 1:
    io.cprint("Train PFNet Generator + Discriminator\n")
    for epoch in range(resume_epoch, opt.niter):
        ep_start_time = time.time()
        if epoch < 30:
            alpha1 = 0.01
            alpha2 = 0.02
        elif epoch < 80:
            alpha1 = 0.05
            alpha2 = 0.1
        else:
            alpha1 = 0.1
            alpha2 = 0.2

        tot_loss, count = 0.0, 0.0  # for AVGin CD Loss per-epoch
        for i, data in enumerate(train_loader, 0):
            real_point, target = data
            B, N, dim = real_point.size()
            it += 1
            count += B

            partials = []
            missing_gts = []
            N_partial_points = N - (opt.crop_point_num * opt.num_holes)
            # Actually cropping + retrieving data!
            for m in range(B):
                cropped, missing = crop_shape(
                    real_point[m], centroids=centroids, n_drop=opt.crop_point_num, n_c=opt.num_holes)

                if cropped.size(0) > N_partial_points:
                    assert opt.num_holes > 1, "Should be no need to resample if not multiple holes"
                    # sampling without replacement
                    choice = torch.randperm(cropped.size(0))[:N_partial_points]
                    cropped = cropped[choice]

                # TODO: PFNet needs to take 2048 points as input!
                #  Method limitation, adding [0, 0, 0] fake vertices
                fake_vertices = torch.zeros(opt.pnum - N_partial_points, 3).float()
                new_partial = torch.cat((cropped, fake_vertices), dim=0)
                new_partial = new_partial[torch.randperm(new_partial.size(0))]  # shuffling
                partials.append(new_partial)
                missing_gts.append(missing)

            # visualizations
            if i == 1:
                # iff. first epoch' it. : save visz of cropped + missing
                visz_folder = os.path.join(save_dir, "train_visz", "epoch_{}".format(epoch))
                if not os.path.exists(visz_folder):
                    os.makedirs(visz_folder)

                # print("ep {} - Saving visualizations into: {}".format(epoch, visz_folder))
                for jj in range(len(partials)):
                    np.savetxt(
                        X=partials[jj].numpy(), fname=os.path.join(visz_folder, '{}_cropped.txt'.format(jj)),
                        fmt='%.5f', delimiter=';')
                    np.savetxt(
                        X=missing_gts[jj].numpy(), fname=os.path.join(visz_folder, '{}_gt.txt'.format(jj)),
                        fmt='%.5f', delimiter=';')

            input_cropped1 = torch.stack(partials)  # [B, 2048, 3]
            real_center = torch.stack(missing_gts)  # [B, 512, 3]

            # logic was: ..
            # p_origin = [0, 0, 0]
            #
            # for m in range(B):
            #     index = random.sample(centroids, 1)  # Random choose one of the viewpoint
            #     distance_list = []
            #     p_center = index[0]
            #     for n in range(opt.pnum):
            #         distance_list.append(distance_squre(real_point[m, 0, n], p_center))
            #
            #     distance_order = sorted(enumerate(distance_list), key=lambda x: x[1])
            #     for sp in range(opt.crop_point_num):
            #         input_cropped1.data[m, 0, distance_order[sp][0]] = torch.FloatTensor([0, 0, 0])
            #         real_center.data[m, 0, sp] = real_point[m, 0, distance_order[sp][0]]

            label.resize_([B, 1]).fill_(real_label)  # [32, 1]
            real_point = real_point.to(device)  # [B, 2048, 3]
            real_center = real_center.to(device)  # [B, 512, 3]
            input_cropped1 = input_cropped1.to(device)  # [B, 2048, 3]: PFNet model needs 2048 points to work..
            label = label.to(device)  # [32, 1]

            ###########################
            # (1) data prepare
            ###########################
            real_center = Variable(real_center, requires_grad=True)
            # real_center = torch.squeeze(real_center, 1)  # [32, 512, 3] - Alli: done before
            real_center_key1_idx = utils.farthest_point_sample(real_center, 64, RAN=False)
            real_center_key1 = utils.index_points(real_center, real_center_key1_idx)  # [32, 64, 3]
            real_center_key1 = Variable(real_center_key1, requires_grad=True)

            real_center_key2_idx = utils.farthest_point_sample(real_center, 128, RAN=True)
            real_center_key2 = utils.index_points(real_center, real_center_key2_idx)
            real_center_key2 = Variable(real_center_key2, requires_grad=True)

            # input_cropped1 = torch.squeeze(input_cropped1, 1)  # [32, 2048, 3] - Alli: done before
            input_cropped2_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[1], RAN=True)
            input_cropped2 = utils.index_points(input_cropped1, input_cropped2_idx)  # [32, 1024, 3]
            input_cropped3_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[2], RAN=False)
            input_cropped3 = utils.index_points(input_cropped1, input_cropped3_idx)  # [32, 512, 3]
            input_cropped1 = Variable(input_cropped1, requires_grad=True)  # [32, 2048, 3]
            input_cropped2 = Variable(input_cropped2, requires_grad=True)  # [32, 1024, 3]
            input_cropped3 = Variable(input_cropped3, requires_grad=True)  # [32, 512, 3]
            input_cropped2 = input_cropped2.to(device)
            input_cropped3 = input_cropped3.to(device)

            if i == 0 and epoch == 0:
                print("-" * 10)
                print("DBG ep %d" % epoch)
                print("cropped1: {};\n".format(str(input_cropped1.size())) +
                      "cropped2: {};\n".format(str(input_cropped2.size())) +
                      "cropped3: {}".format(str(input_cropped3.size())))
                print("real_center: {}\n".format(str(real_center.size())) +
                      "real_center_key1: {}\n".format(str(real_center_key1.size())) +
                      "real_center_key2: {}\n".format(str(real_center_key2.size())))
                print("-" * 10 + '\n')

            input_cropped = [
                input_cropped1, input_cropped2, input_cropped3]
            # input_cropped: [ [B,2048,3], [B,1024,3], [B,512,3] ]
            point_netG = point_netG.train()
            point_netD = point_netD.train()

            ###########################
            # (2) Update D network
            ###########################
            point_netD.zero_grad()
            real_center = torch.unsqueeze(real_center, 1)  # [32, 1, 512, 3]
            output = point_netD(real_center)
            errD_real = criterion(output, label)  # real points + real label: provided here
            errD_real.backward()

            fake_center1, fake_center2, fake = point_netG(input_cropped)  # (input_cropped) has len==3
            fake = torch.unsqueeze(fake, 1)
            '''
            fake: [32, 1, 512, 3]
            fake_center1: [32, 64, 3]
            fake_center2: [32, 128, 3]
            '''
            label.data.fill_(fake_label)
            output = point_netD(fake.detach())
            errD_fake = criterion(output, label)
            errD_fake.backward()
            errD = errD_real + errD_fake
            optimizerD.step()

            #######################################################
            # (3) Update G network: maximize log(D(G(z)))
            #######################################################
            point_netG.zero_grad()
            label.data.fill_(real_label)  # foolish
            output = point_netD(fake)
            errG_D = criterion(output, label)
            errG_l2 = 0

            fake = fake.squeeze(1).contiguous()  # [32, 1, 512, 3] -> [32, 512, 3]
            real_center = real_center.squeeze(1).contiguous()
            # print("dbg fake: ", fake.size())
            # print("dbg real_center: ", real_center.size())
            assert fake.size() == real_center.size(), "fail fake shape"
            d1, d2, _, _ = NND.nnd(fake, real_center)
            CD_LOSS = 100 * (0.5 * torch.mean(d1) + 0.5 * torch.mean(d2))

            # computing also errG_l2
            ''' fake center 1 '''
            fake_center1 = fake_center1.contiguous()
            real_center_key1 = real_center_key1.contiguous()
            # print("dbg fake_center1: ", fake_center1.size())
            # print("dbg real_center_key1: ", real_center_key1.size())
            assert fake_center1.size() == real_center_key1.size(), "fail fake 1 {}".format(str(fake_center1.size()))
            d1, d2, _, _ = NND.nnd(fake_center1, real_center_key1)
            cd_fake_1 = 100 * (0.5 * torch.mean(d1) + 0.5 * torch.mean(d2))

            ''' fake center 2 '''
            fake_center2 = fake_center2.contiguous()
            real_center_key2 = real_center_key2.contiguous()
            # print("dbg fake_center2: ", fake_center2.size())
            # print("dbg real_center_key2: ", real_center_key2.size())
            assert fake_center2.size() == real_center_key2.size(), "fail fake 2 {}".format(str(fake_center2.size()))
            d1, d2, _, _ = NND.nnd(fake_center2, real_center_key2)
            cd_fake_2 = 100 * (0.5 * torch.mean(d1) + 0.5 * torch.mean(d2))

            errG_l2 = CD_LOSS + alpha1 * cd_fake_1 + alpha2 * cd_fake_2  # same but more efficient!
            # CD_LOSS = criterion_PointLoss(torch.squeeze(fake, 1), torch.squeeze(real_center, 1))
            # errG_l2 = criterion_PointLoss(torch.squeeze(fake, 1), torch.squeeze(real_center, 1)) \
            #           + alpha1 * criterion_PointLoss(fake_center1, real_center_key1) \
            #           + alpha2 * criterion_PointLoss(fake_center2, real_center_key2)

            errG = (1 - opt.wtl2) * errG_D + opt.wtl2 * errG_l2
            errG.backward()
            optimizerG.step()

            if i == 1:
                # curr. epoch first train it.
                # save visz generated (fake) missing part
                print("dbg fake: ", fake.size())
                assert (visz_folder is not None and os.path.exists(visz_folder))
                fake = fake.cpu().detach().data.numpy()
                for jj in range(len(fake)):
                    np.savetxt(
                        X=fake[jj], fname=os.path.join(visz_folder, '{}_fake.txt'.format(jj)), fmt='%.5f', delimiter=';')

            if it % 10 == 0:
                io.cprint(
                    '[%d/%d][%d/%d] Loss_D: %.4f, errG_D: %.4f, errG_l2: %.4f, errG: %.4f, CD_LOSS: %.4f' %
                    (epoch, opt.niter, i, len(train_loader),
                     errD.data, errG_D.data,  # discriminator part losses
                     errG_l2, errG, CD_LOSS  # generator part losses
                     )
                )

            tot_loss += CD_LOSS.item() * B

        schedulerD.step()
        schedulerG.step()
        print('[%d/%d] Epoch Train CD Loss: %.5f' % (epoch, opt.niter, tot_loss * 1.0 / count))
        tb.add_scalar('Train/CD_LOSS', tot_loss * 1.0 / count, epoch)

        if epoch % 10 == 0:

            # checkpoint
            torch.save(
                {'epoch': epoch + 1,
                 'state_dict': point_netG.module.state_dict() if isinstance(point_netG, torch.nn.DataParallel) else point_netG.state_dict(),
                 'optimizer': optimizerG.state_dict(),
                 'scheduler': schedulerG.state_dict(),
                 },
                os.path.join(point_netG_saving, 'gen_' + str(epoch) + '.pth')
            )
            torch.save(
                {'epoch': epoch + 1,
                 'state_dict': point_netD.module.state_dict() if isinstance(point_netD, torch.nn.DataParallel) else point_netD.state_dict(),
                 'optimizer': optimizerD.state_dict(),
                 'scheduler': schedulerD.state_dict(),
                 },
                os.path.join(point_netD_saving, 'discr_' + str(epoch) + '.pth'))

            # Evaluate model on test set
            sum_cd_test, count_test = 0.0, 0.0
            for i, data in enumerate(test_dataloader, 0):
                real_point, target = data
                B, N, dim = real_point.size()
                count_test += B

                partials = []
                missing_gts = []
                N_partial_points = N - (opt.crop_point_num * opt.num_holes)

                # Actually cropping + retrieving data!
                for m in range(B):
                    cropped, missing = crop_shape(
                        real_point[m], centroids=centroids, n_drop=opt.crop_point_num, n_c=opt.num_holes)

                    if cropped.size(0) > N_partial_points:
                        assert opt.num_holes > 1, "No need to resample if not multiple holes"
                        # re-sampling WITHOUT replacement is needed!
                        choice = torch.randperm(cropped.size(0))[:N_partial_points]
                        cropped = cropped[choice]

                    # TODO: PFNet needs to take 2048 points as input! Method limitation! Adding [0, 0, 0] fake vertices
                    fake_vertices = torch.zeros(opt.pnum - N_partial_points, 3).float()
                    new_partial = torch.cat((cropped, fake_vertices), dim=0)
                    new_partial = new_partial[  # shuffling
                        torch.randperm(new_partial.size(0))]
                    partials.append(new_partial)
                    missing_gts.append(missing)

                input_cropped1 = torch.stack(partials)  # [B, 2048, 3]
                real_center = torch.stack(missing_gts)  # [B, 512, 3]
                real_point = real_point.to(device)  # [B, 2048, 3]
                real_center = real_center.to(device)  # [B, 512, 3]
                input_cropped1 = input_cropped1.to(device)  # [B, 2048, 3]: PFNet model needs 2048 points to work..

                ###########################
                # (1) data prepare
                ###########################
                input_cropped2_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[1], RAN=True)
                input_cropped2 = utils.index_points(input_cropped1, input_cropped2_idx)  # [32, 1024, 3]
                input_cropped3_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[2], RAN=False)
                input_cropped3 = utils.index_points(input_cropped1, input_cropped3_idx)  # [32, 512, 3]
                input_cropped1 = Variable(input_cropped1, requires_grad=False)  # [32, 2048, 3]
                input_cropped2 = Variable(input_cropped2, requires_grad=False)  # [32, 1024, 3]
                input_cropped3 = Variable(input_cropped3, requires_grad=False)  # [32, 512, 3]
                input_cropped2 = input_cropped2.to(device)
                input_cropped3 = input_cropped3.to(device)
                input_cropped = [input_cropped1, input_cropped2, input_cropped3]

                point_netG.eval()
                with torch.no_grad():
                    _, _, fake = point_netG(input_cropped)

                assert fake.size() == real_center.size()
                fake = fake.contiguous()
                real_center = real_center.contiguous()
                d1, d2, _, _ = NND.nnd(fake, real_center)
                cd_test_loss = 100 * (0.5 * torch.mean(d1) + 0.5 * torch.mean(d2))
                sum_cd_test += cd_test_loss * B

            test_loss = sum_cd_test * 1.0 / count_test
            io.cprint('Ep Test [%d/%d] Loss: %.5f ' % (
                epoch, opt.niter, test_loss), color="b")
            tb.add_scalar('Test/CD_LOSS', test_loss, epoch)

            is_best = test_loss < best_test
            best_test = min(best_test, test_loss)
            if is_best:
                # best model case
                best_ep = epoch
                io.cprint("New best test %.5f at epoch %d" % (best_test, best_ep))
                shutil.copyfile(
                    src=os.path.join(point_netG_saving, 'gen_' + str(epoch) + '.pth'),
                    dst=os.path.join(point_netG_saving, 'best_gen.pth'),
                )

        io.cprint('[%d/%d] - Elapsed Time: {}\n'.format(
            time.strftime("%M:%S", time.gmtime(time.time() - ep_start_time))) % (epoch, opt.niter))
else:
    io.cprint("Train PFNet Generator (without Discriminator)\n")
    for epoch in range(resume_epoch, opt.niter):
        ep_start_time = time.time()
        if epoch < 30:
            alpha1 = 0.01
            alpha2 = 0.02
        elif epoch < 80:
            alpha1 = 0.05
            alpha2 = 0.1
        else:
            alpha1 = 0.1
            alpha2 = 0.2

        tot_loss, count = 0.0, 0.0  # for AVGin CD Loss per-epoch
        for i, data in enumerate(train_loader, 0):
            real_point, target = data
            B, N, dim = real_point.size()
            it += 1
            count += B

            partials = []
            missing_gts = []
            N_partial_points = N - (opt.crop_point_num * opt.num_holes)
            # Actually cropping + retrieving data!
            for m in range(B):
                cropped, missing = crop_shape(
                    real_point[m], centroids=centroids, n_drop=opt.crop_point_num, n_c=opt.num_holes)

                if cropped.size(0) > N_partial_points:
                    assert opt.num_holes > 1, "Should be no need to resample if not multiple holes"
                    # without replacement
                    choice = torch.randperm(cropped.size(0))[:N_partial_points]
                    cropped = cropped[choice]

                # PFNet limitation: network must take 2048 points in input! Adding [0, 0, 0] fake vertices to fix
                fake_vertices = torch.zeros(opt.pnum - N_partial_points, 3).float()
                new_partial = torch.cat((cropped, fake_vertices), dim=0)
                new_partial = new_partial[torch.randperm(new_partial.size(0))]  # shuffling
                partials.append(new_partial)
                missing_gts.append(missing)

            # visualizations
            if i == 1 and epoch % 10 == 0:
                visz_folder = os.path.join(save_dir, "train_visz", "epoch_{}".format(epoch))
                if not os.path.exists(visz_folder):
                    os.makedirs(visz_folder)

                for jj in range(len(partials)):
                    np.savetxt(
                        X=partials[jj].numpy(), fname=os.path.join(visz_folder, '{}_cropped.txt'.format(jj)),
                        fmt='%.5f', delimiter=';')
                    np.savetxt(
                        X=missing_gts[jj].numpy(), fname=os.path.join(visz_folder, '{}_gt.txt'.format(jj)),
                        fmt='%.5f', delimiter=';')

            input_cropped1 = torch.stack(partials)  # [B, 2048, 3]
            real_center = torch.stack(missing_gts)  # [B, 512, 3]
            real_point = real_point.to(device)  # [B, 2048, 3]
            real_center = real_center.to(device)  # [B, 512, 3]
            input_cropped1 = input_cropped1.to(device)  # [B, 2048, 3]: PFNet model needs 2048 points to work..

            ###########################
            # (1) data prepare
            ###########################
            real_center = Variable(real_center, requires_grad=True)
            real_center_key1_idx = utils.farthest_point_sample(real_center, 64, RAN=False)
            real_center_key1 = utils.index_points(real_center, real_center_key1_idx)  # [32, 64, 3]
            real_center_key1 = Variable(real_center_key1, requires_grad=True)

            real_center_key2_idx = utils.farthest_point_sample(real_center, 128, RAN=True)
            real_center_key2 = utils.index_points(real_center, real_center_key2_idx)
            real_center_key2 = Variable(real_center_key2, requires_grad=True)

            input_cropped2_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[1], RAN=True)
            input_cropped2 = utils.index_points(input_cropped1, input_cropped2_idx)  # [32, 1024, 3]
            input_cropped3_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[2], RAN=False)
            input_cropped3 = utils.index_points(input_cropped1, input_cropped3_idx)  # [32, 512, 3]
            input_cropped1 = Variable(input_cropped1, requires_grad=True)  # [32, 2048, 3]
            input_cropped2 = Variable(input_cropped2, requires_grad=True)  # [32, 1024, 3]
            input_cropped3 = Variable(input_cropped3, requires_grad=True)  # [32, 512, 3]
            input_cropped2 = input_cropped2.to(device)
            input_cropped3 = input_cropped3.to(device)

            if i == 0 and epoch == 0:
                print("-" * 10)
                print("DBG ep %d" % epoch)
                print("cropped1: {};\n".format(str(input_cropped1.size())) +
                      "cropped2: {};\n".format(str(input_cropped2.size())) +
                      "cropped3: {}".format(str(input_cropped3.size())))
                print("real_center: {}\n".format(str(real_center.size())) +
                      "real_center_key1: {}\n".format(str(real_center_key1.size())) +
                      "real_center_key2: {}\n".format(str(real_center_key2.size())))
                print("-" * 10 + '\n')

            input_cropped = [input_cropped1, input_cropped2, input_cropped3]  # [ [32,2048,3], [32,1024,3], [32,512,3] ]
            point_netG = point_netG.train()
            point_netG.zero_grad()
            fake_center1, fake_center2, fake = point_netG(input_cropped)

            # compute chamfer on missing part final prediction
            fake = fake.contiguous()  # [32, 512, 3]
            real_center = real_center.contiguous()  # [32, 512, 3]
            assert fake.size() == real_center.size(), "fail fake shape"
            d1, d2, _, _ = NND.nnd(fake, real_center)
            CD_LOSS = 100 * (0.5 * torch.mean(d1) + 0.5 * torch.mean(d2))

            # computing also errG_l2
            ''' fake center 1 '''
            fake_center1 = fake_center1.contiguous()
            real_center_key1 = real_center_key1.contiguous()
            assert fake_center1.size() == real_center_key1.size(), "fail fake 1 {}".format(str(fake_center1.size()))
            d1, d2, _, _ = NND.nnd(fake_center1, real_center_key1)
            cd_fake_1 = 100 * (0.5 * torch.mean(d1) + 0.5 * torch.mean(d2))

            ''' fake center 2 '''
            fake_center2 = fake_center2.contiguous()
            real_center_key2 = real_center_key2.contiguous()
            assert fake_center2.size() == real_center_key2.size(), "fail fake 2 {}".format(str(fake_center2.size()))
            d1, d2, _, _ = NND.nnd(fake_center2, real_center_key2)
            cd_fake_2 = 100 * (0.5 * torch.mean(d1) + 0.5 * torch.mean(d2))

            errG_l2 = CD_LOSS + alpha1 * cd_fake_1 + alpha2 * cd_fake_2
            errG_l2.backward()
            optimizerG.step()

            if i == 1 and epoch % 10 == 0:
                print("fake: ", fake.size())
                assert (visz_folder is not None and os.path.exists(visz_folder))
                fake = fake.cpu().detach().data.numpy()
                for jj in range(len(fake)):
                    np.savetxt(
                        X=fake[jj], fname=os.path.join(visz_folder, '{}_fake.txt'.format(jj)), fmt='%.5f',
                        delimiter=';')

            if it % 10 == 0:
                io.cprint(
                    '[%d/%d][%d/%d] errG_l2: %.4f, CD_LOSS: %.4f' %
                    (epoch, opt.niter, i, len(train_loader), errG_l2, CD_LOSS))
            tot_loss += CD_LOSS.item() * B

        schedulerG.step()
        print('[%d/%d] Epoch Train CD Loss: %.5f' % (epoch, opt.niter, tot_loss * 1.0 / count))
        tb.add_scalar('Train/CD_LOSS', tot_loss * 1.0 / count, epoch)

        if epoch % 10 == 0:

            # checkpoint
            torch.save(
                {'epoch': epoch + 1,
                 'state_dict': point_netG.module.state_dict() if isinstance(point_netG,
                                                                            torch.nn.DataParallel) else point_netG.state_dict(),
                 'optimizer': optimizerG.state_dict(),
                 'scheduler': schedulerG.state_dict(),
                 },
                os.path.join(point_netG_saving, 'gen_' + str(epoch) + '.pth')
            )

            # Evaluate model on test set
            sum_cd_test, count_test = 0.0, 0.0
            for i, data in enumerate(test_dataloader, 0):
                real_point, target = data
                B, N, dim = real_point.size()
                count_test += B

                partials = []
                missing_gts = []
                N_partial_points = N - (opt.crop_point_num * opt.num_holes)
                # Actually cropping + retrieving data!
                for m in range(B):
                    cropped, missing = crop_shape(
                        real_point[m], centroids=centroids, n_drop=opt.crop_point_num, n_c=opt.num_holes)

                    if cropped.size(0) > N_partial_points:
                        assert opt.num_holes > 1
                        choice = torch.randperm(cropped.size(0))[:N_partial_points]
                        cropped = cropped[choice]

                    fake_vertices = torch.zeros(opt.pnum - N_partial_points, 3).float()
                    new_partial = torch.cat((cropped, fake_vertices), dim=0)
                    new_partial = new_partial[  # shuffling
                        torch.randperm(new_partial.size(0))]
                    partials.append(new_partial)
                    missing_gts.append(missing)

                input_cropped1 = torch.stack(partials)  # [B, 2048, 3]
                real_center = torch.stack(missing_gts)  # [B, 512, 3]
                real_point = real_point.to(device)  # [B, 2048, 3]
                real_center = real_center.to(device)  # [B, 512, 3]
                input_cropped1 = input_cropped1.to(device)  # [B, 2048, 3]: PFNet model needs 2048 points to work..

                ###########################
                # (1) data prepare
                ###########################
                input_cropped2_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[1], RAN=True)
                input_cropped2 = utils.index_points(input_cropped1, input_cropped2_idx)  # [32, 1024, 3]
                input_cropped3_idx = utils.farthest_point_sample(input_cropped1, opt.point_scales_list[2], RAN=False)
                input_cropped3 = utils.index_points(input_cropped1, input_cropped3_idx)  # [32, 512, 3]
                input_cropped1 = Variable(input_cropped1, requires_grad=False)  # [32, 2048, 3]
                input_cropped2 = Variable(input_cropped2, requires_grad=False)  # [32, 1024, 3]
                input_cropped3 = Variable(input_cropped3, requires_grad=False)  # [32, 512, 3]
                input_cropped2 = input_cropped2.to(device)
                input_cropped3 = input_cropped3.to(device)
                input_cropped = [input_cropped1, input_cropped2, input_cropped3]

                point_netG.eval()
                with torch.no_grad():
                    _, _, fake = point_netG(input_cropped)

                assert fake.size() == real_center.size()
                fake = fake.contiguous()
                real_center = real_center.contiguous()
                d1, d2, _, _ = NND.nnd(fake, real_center)
                cd_test_loss = 100 * (0.5 * torch.mean(d1) + 0.5 * torch.mean(d2))
                sum_cd_test += cd_test_loss * B

            test_loss = sum_cd_test * 1.0 / count_test
            io.cprint('Ep Test [%d/%d] Loss: %.5f ' % (epoch, opt.niter, test_loss), color="b")
            tb.add_scalar('Test/CD_LOSS', test_loss, epoch)
            is_best = test_loss < best_test
            best_test = min(best_test, test_loss)

            # TODO: here assuming that test it. is same of save it.
            if is_best:
                best_ep = epoch
                io.cprint("New best test %.5f at epoch %d" % (best_test, best_ep))
                shutil.copyfile(
                    src=os.path.join(point_netG_saving, 'gen_' + str(epoch) + '.pth'),
                    dst=os.path.join(point_netG_saving, 'best_gen.pth')
                )

        io.cprint('[%d/%d] - Elapsed Time: {}\n'.format(
            time.strftime("%M:%S", time.gmtime(time.time() - ep_start_time))) % (epoch, opt.niter))

io.cprint("Best test %.6f at epoch %d " % (best_test, best_ep))
io.cprint('-'*30)

