import open3d as o3
import argparse
import os
import os.path as osp
import json
import time
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from models.GPDNet import GPDLocalFE
import torch_nndistance as NND
from utils import IOStream, safe_make_dirs
from pretext_training.pretext_data import PretextDataset
from tqdm import tqdm
silent_warn = o3

"""
Training example:
    python pretrain_local_denoise.py --batch_size 24 --config ./pretext_training/local_GPD_pretrain_config.json
     --exp_name gpd_mse_den
"""


def parse_args():
    parser = argparse.ArgumentParser(description='GPD Denoise pretext trainer')
    parser.add_argument('--config', type=str,
                        default="./pretext_training/local_GPD_pretrain_config.json",
                        help='path to configuration file for local denoising pretext architecture')
    parser.add_argument('--epochs', type=int, default=101)
    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--workers', type=int, default=6, help='num of workers loading data for each dataloader')
    parser.add_argument('--checkpoints_dir', type=str, default='pretext_weights/new_pretrainings/local_denoising')
    parser.add_argument('--exp_name', type=str, default='mse_GPD_pretrain')
    parser.add_argument('--data_root', type=str,
                        default='/home/antonioa/data/shapenetcore_partanno_segmentation_benchmark_v0')
    parser.add_argument('--parallel', action='store_true', help="enables pytorch DataParallel training")
    parser.add_argument('--class_choice',
                        default="Airplane,Bag,Cap,Car,Chair,Guitar,Lamp,Laptop,"
                                "Motorbike,Mug,Pistol,Skateboard,Table",
                        help='Classes to train on: default is 13 classes used in PF-Net')
    return parser.parse_args()


def main(opt):
    exp_dir = osp.join(opt.checkpoints_dir, opt.exp_name)
    tb_dir, models_dir = osp.join(exp_dir, "tb_logs"), osp.join(exp_dir, "models")
    safe_make_dirs([tb_dir, models_dir])
    io = IOStream(osp.join(exp_dir, "log.txt"))
    tb_logger = SummaryWriter(logdir=tb_dir)
    assert os.path.exists(opt.config), "wrong config path"
    with open(opt.config) as cf:
        config = json.load(cf)
    io.cprint(f"Arguments: {str(opt)}")
    io.cprint(f"Config: {str(config)} \n")

    if len(opt.class_choice) > 0:
        class_choice = ''.join(opt.class_choice.split()).split(",")  # sanitize + split(",")
        io.cprint("Class choice: {}".format(str(class_choice)))
    else:
        class_choice = None

    train_dataset = PretextDataset(
        root=opt.data_root, task='denoise', class_choice=class_choice, npoints=config["num_points"],
        split='train', normalize=True, noise_mean=config["noise_mean"], noise_std=config["noise_std"])

    test_dataset = PretextDataset(
        root=opt.data_root, task='denoise', class_choice=class_choice, npoints=config["num_points"],
        split='test', normalize=True, noise_mean=config["noise_mean"], noise_std=config["noise_std"])

    train_loader = DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True, drop_last=True, num_workers=opt.workers)

    test_loader = DataLoader(
        test_dataset, batch_size=opt.batch_size, shuffle=False, drop_last=False, num_workers=opt.workers)

    criterion = nn.MSELoss()  # loss function for denoising
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # MODEL
    model = GPDLocalFE(config)
    if opt.parallel:
        io.cprint(f"DataParallel training with {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    model = model.to(device)
    io.cprint(f'model: {str(model)}')

    # OPTIMIZER + SCHEDULER
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    train_start = time.time()
    for epoch in range(opt.epochs):
        # TRAIN
        # we compute both MSE and Chamfer Distance distances between the cleaned pointcloud and the clean GT,
        # where cleaned = model(noised)
        # .. Anyway MSE is used as loss function and Chamfer Distance is just an additional metric
        ep_start = time.time()
        train_mse, train_cd = train_one_epoch(train_loader, model, optimizer, criterion, device)
        train_time = time.strftime("%M:%S", time.gmtime(time.time() - ep_start))
        io.cprint("Train %d, time: %s, MSE (loss): %.6f, CD (dist): %.6f" % (
            epoch, train_time, train_mse, train_cd))
        tb_logger.add_scalar("Train/MSE_loss", train_mse, epoch)
        tb_logger.add_scalar("Train/CD_dist", train_cd, epoch)

        # TEST
        mse_test, cd_test = test(test_loader, model, criterion, device)
        io.cprint("Test %d, MSE (loss): %.6f, CD (dist): %.6f" % (epoch, mse_test, cd_test))
        tb_logger.add_scalar("Test/MSE", mse_test, epoch)
        tb_logger.add_scalar("Test/CD", cd_test, epoch)

        # LR SCHEDULING
        scheduler.step()

        if epoch % 10 == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict() if not opt.parallel else model.module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                }, osp.join(models_dir, "local_denoise_{}.pth".format(epoch)))

    hours, rem = divmod(time.time() - train_start, 3600)
    minutes, seconds = divmod(rem, 60)
    io.cprint("Training ended in {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))


def train_one_epoch(train_loader, model, optimizer, criterion, device):
    model.train()
    tot_loss = 0
    tot_cd_dist = 0
    # TODO: remove unnecessary dependencies (tqdm)
    for i, data in tqdm(enumerate(train_loader, 0), total=len(train_loader),
                        smoothing=0.9, desc='train', dynamic_ncols=True):
        assert len(data) == 3, 'train: expected tuple: (noised, clean, cls)'
        noised, clean, _ = data
        bs = len(noised)
        noised = noised.to(device)  # [bs, npoints, 3]
        clean = clean.to(device)  # [bs, npoints, 3]
        cleaned = model(noised)
        assert cleaned.size() == clean.size()
        loss = criterion(cleaned, clean)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluate also CD distance
        cleaned = cleaned.contiguous()
        clean = clean.contiguous()
        dist1, dist2, _, _ = NND.nnd(cleaned, clean)
        cd_dist = 50 * torch.mean(dist1) + 50 * torch.mean(dist2)

        tot_loss += loss.item() * bs  # MSE Loss
        tot_cd_dist += cd_dist.item() * bs  # Chamfer Distance

    tot_loss = tot_loss * 1.0 / len(train_loader.dataset)
    tot_cd_dist = tot_cd_dist * 1.0 / len(train_loader.dataset)
    return tot_loss, tot_cd_dist


def test(test_loader, model, criterion, device):
    model.eval()
    tot_loss = 0  # MSE
    tot_cd_dist = 0  # Chamfer dist
    # TODO: remove unnecessary dependencies (tqdm)
    for i, data in tqdm(enumerate(test_loader, 0), total=len(test_loader),
                        smoothing=0.9, desc='test', dynamic_ncols=True):
        assert len(data) == 3
        noised, clean, _ = data
        bs = len(noised)
        noised = noised.to(device)  # [bs, npoints, 3]
        clean = clean.to(device)

        with torch.no_grad():
            cleaned = model(noised)
            assert cleaned.size() == clean.size()
            loss = criterion(cleaned, clean)
            tot_loss += loss.item() * bs

            # evaluate also CD distance
            cleaned = cleaned.contiguous()
            clean = clean.contiguous()
            dist1, dist2, _, _ = NND.nnd(cleaned, clean)
            cd_dist = 50 * torch.mean(dist1) + 50 * torch.mean(dist2)
            tot_cd_dist += cd_dist.item() * bs

    tot_loss = tot_loss * 1.0 / len(test_loader.dataset)
    tot_cd_dist = tot_cd_dist * 1.0 / len(test_loader.dataset)
    return tot_loss, tot_cd_dist


if __name__ == '__main__':
    args = parse_args()
    main(args)
