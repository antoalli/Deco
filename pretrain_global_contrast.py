import open3d as o3
import argparse
import os
import os.path as osp
import sys
import time
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pretext_training.pretext_data import PretextDataset
from pretext_training.supcon_loss import SimCLRLoss, SupConLoss
from pretext_training.data_utils import *
from model_utils import GlobalFeat, Projector
from utils import IOStream, safe_make_dirs
from tensorboardX import SummaryWriter
from tqdm import tqdm

silent_warn = o3

"""
Training example:
4 Positive Samples per-shape: 
    python pretrain_global_contrast.py --exp_name simCLR_cropRot_ScaleP05_Jitt_sgdCos_DP_4POS --batch_size 40 --parallel
     --use_sgd --lr 0.5 --sched cos --temp 0.5 --projection_dim 128 --num_positive_samples 4

Alt. 2 - Simple SimCLR only two positive samples:
    python pretrain_global_contrast.py --exp_name simCLR_cropRJitt_sgdCos_DP --batch_size 76 --parallel
     --use_sgd --lr 0.5 --sched cos --temp 0.5 --projection_dim 128
"""


def parse_args():
    parser = argparse.ArgumentParser(description='Global Contrastive pretext trainer')

    # Experiment
    parser.add_argument('--checkpoints_dir', type=str, default='pretext_weights/new_pretrainings/global_contrastive')
    parser.add_argument('--exp_name', type=str, default='simCLR_exp')
    parser.add_argument('--epochs', type=int, default=251, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=40, help='batch_size')
    parser.add_argument('--workers', type=int, default=8, help='num of workers')
    parser.add_argument('--parallel', action='store_true', help="enables pytorch DataParallel training")

    # Keep these as deafult!
    parser.add_argument('--nearest_neighboors', type=int, default=24)
    parser.add_argument('--latent_dim', type=int, default=1024)
    parser.add_argument('--num_points', type=int, default=2048)

    # Dataset
    parser.add_argument('--data_root', type=str,
                        default="/home/antonioa/data/shapenetcore_partanno_segmentation_benchmark_v0")
    parser.add_argument('--class_choice', type=str,
                        default="Airplane,Bag,Cap,Car,Chair,Guitar,Lamp,Laptop,Motorbike,Mug,Pistol,Skateboard,Table")

    # Optimization
    parser.add_argument('--use_sgd', action='store_true')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--sched', type=str, default='',
                        choices=['step', 'cos', ''], help='choose a lr scheduler')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='only with cos scheduler!')

    # Contrastive params
    parser.add_argument('--temp', '-T', type=float, default=0.5,
                        help='temperature for SimCLR loss function')
    parser.add_argument('--projection_dim', '-P', type=int, default=128)
    parser.add_argument('--num_positive_samples', '-PS', type=int, default=2)

    return parser.parse_args()


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


args = parse_args()
exp_dir = os.path.join(args.checkpoints_dir, args.exp_name + '_' + str(int(time.time())))
tb_dir, models_dir = osp.join(exp_dir, "tb_logs"), osp.join(exp_dir, "models")
safe_make_dirs([tb_dir, models_dir])
io = IOStream(osp.join(exp_dir, "log.txt"))
io.cprint(f"Arguments: {str(args)} \n")
tb_writer = SummaryWriter(logdir=tb_dir)
centroids = np.asarray([[1, 0, 0], [0, 0, 1], [1, 0, 1], [-1, 0, 0], [-1, 1, 0]])  # same as PFNet

if args.num_positive_samples > 2:
    criterion = SupConLoss(temperature=args.temp, base_temperature=1, contrast_mode='all')
else:
    criterion = SimCLRLoss(temperature=args.temp)

io.cprint("Contrastive learning params: ")
io.cprint(f"criterion: {str(criterion)}")
io.cprint(f"num positive samples: {args.num_positive_samples}")
io.cprint(f"centroids cropping: {str(centroids)}")

train_transforms = transforms.Compose(
    [Pointcloud2Partial(centroids=centroids, crop_point_num=512),
     PointcloudRotate(),
     transforms.RandomApply([PointcloudScale()], p=0.5),
     PointcloudJitter()])

class_choice = args.class_choice
if len(class_choice) > 0:
    class_choice = ''.join(class_choice.split()).split(",")  # list
else:
    class_choice = None

train_dataset = PretextDataset(
    root=args.data_root,
    task='contrast',
    class_choice=class_choice,
    npoints=args.num_points,
    split='train',
    centroids=centroids,
    crop_point_num=512,
    num_positive_samples=args.num_positive_samples,
    transforms=train_transforms
)

io.cprint("transforms: \n" + str(train_transforms) + '\n')

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=args.workers,
    pin_memory=True
)

num_classes = len(train_dataset.classes.keys())
io.cprint('Num classes: {}'.format(num_classes))

encoder = GlobalFeat(
    k=args.nearest_neighboors,
    emb_dims=args.latent_dim
)

encoder.apply(weights_init_normal)

projector = Projector(
    in_feat=args.latent_dim,
    projection_dim=args.projection_dim
)

projector.apply(weights_init_normal)

if args.parallel:
    assert torch.cuda.device_count() > 1
    encoder = torch.nn.DataParallel(encoder)
    projector = torch.nn.DataParallel(projector)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = encoder.to(device)
projector = projector.to(device)

io.cprint("encoder:\n " + str(encoder) + '\n')
io.cprint("projector:\n " + str(projector) + '\n')

if not args.use_sgd:
    # adam - lr should be 0.001 (1.0e-3; 3.0e-3)
    io.cprint("adam optimizer - lr: %.6f, weight_decay: %.6f " % (args.lr, args.weight_decay))
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay)
else:
    # SGD + CosineAnnealing - configuration from SupContast: base_lr = 0.5
    io.cprint("SGD optimizer - lr: %.6f, weight_decay: %.6f " % (args.lr, args.weight_decay))
    optimizer = torch.optim.SGD(
        list(encoder.parameters()) + list(projector.parameters()),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)

if args.sched == 'step':
    io.cprint("lr step scheduling")
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
elif args.sched == 'cos':
    io.cprint("lr cosine scheduling")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, args.epochs, eta_min=args.lr * (args.lr_decay_rate ** 3))
else:
    # no scheduler
    scheduler = None

start_epoch = -1


def train_epoch(epoch):
    ep_start = time.time()
    encoder.train()
    projector.train()
    tot_loss = 0
    count = 0

    for i, (data, cls) in tqdm(enumerate(train_loader, 0), total=len(train_loader),
                               smoothing=0.9, desc='train', dynamic_ncols=True):
        assert len(data) == args.num_positive_samples
        bs = cls.shape[0]
        count += bs

        if i == 0:
            io.cprint('it %d dbg batch cat distr.: ' % i + str(cls))

        proj_feat = []
        for k in range(len(data)):
            # data[k] is view num-k of all shapes in batch
            x_k = data[k].to(device).float().permute(0, 2, 1)  # [B, 3, N]
            # feat encoding
            h_k = encoder(x_k)
            # projection + normalization
            z_k = F.normalize(projector(h_k), dim=1).unsqueeze(1)  # [B, 1, 128]
            proj_feat.append(z_k)
        proj_feat = torch.cat(proj_feat, dim=1)  # [B, num_positive_samples, 128]
        loss = criterion(proj_feat)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        tot_loss += loss.item() * bs

    tot_loss = tot_loss * 1.0 / count
    io.cprint(
        "Train %d, time: %s, Contr. loss: %.6f" % (
            epoch, time.strftime("%M:%S", time.gmtime(time.time() - ep_start)), tot_loss))

    tb_writer.add_scalar("Train/Loss", tot_loss, epoch)

    if epoch % 10 == 0:
        torch.save({
            'epoch': epoch,
            'global_encoder': encoder.module.state_dict() if isinstance(encoder, torch.nn.DataParallel)
            else encoder.state_dict(),
            'projector': projector.module.state_dict() if isinstance(projector, torch.nn.DataParallel)
            else projector.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler is not None else None,
            'loss': tot_loss
        },
            osp.join(models_dir, 'global_contrastive_' + str(epoch) + '.pth'))
    return tot_loss


if __name__ == '__main__':
    start_time = time.time()
    tot_epochs = args.epochs
    min_loss = sys.float_info.max
    best_epoch = -1
    io.cprint('Num train Epochs: {}'.format(tot_epochs))
    for curr_epoch in range(start_epoch + 1, tot_epochs + 1):
        ep_loss = train_epoch(curr_epoch)
        if scheduler:
            scheduler.step()  # step after epoch
        is_best = ep_loss < min_loss
        min_loss = min(ep_loss, min_loss)
        if is_best:
            best_epoch = curr_epoch
            io.cprint("** new min loss: %.6f at epoch %d **" % (min_loss, curr_epoch))
            # save best model
            torch.save({
                'epoch': curr_epoch,
                'global_encoder': encoder.module.state_dict() if isinstance(encoder, torch.nn.DataParallel)
                else encoder.state_dict(),
                'projector': projector.module.state_dict() if isinstance(projector, torch.nn.DataParallel)
                else projector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'loss': min_loss
            },
                osp.join(models_dir, 'best_model.pth'))

    hours, rem = divmod(time.time() - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    io.cprint("## Training time {:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds))
    io.cprint("## Min loss %.6f at epoch %d" % (min_loss, best_epoch))
