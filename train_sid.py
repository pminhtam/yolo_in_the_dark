import torch
import argparse
from torch.utils.data import DataLoader
from loss import losses_denoise as losses
import os

# import h5py
from data.dataset_denoise.data_provider_city import SingleLoader
import torch.optim as optim
import numpy as np
# import model
from utils.metric import calculate_psnr
from model.sid import Unet
from model.encoder_g2e import Encoderg2e
from utils.training_util import save_checkpoint,MovingAverage, load_checkpoint
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
def train(args):
    torch.set_num_threads(args.num_workers)
    torch.manual_seed(0)
    model = Unet()
    model_encoder = Encoderg2e(in_channels=3)
    data_set = SingleLoader(noise_dir=args.noise_dir, gt_dir=args.gt_dir, image_size=args.image_size)

    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_func = losses.CharbonnierLoss().to(device)
    # loss_func = losses.AlginLoss().to(device)
    checkpoint_dir = args.checkpoint
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr
    )
    optimizer_encoder = optim.Adam(
        model_encoder.parameters(),
        lr=args.lr
    )
    optimizer.zero_grad()
    optimizer_encoder.zero_grad()
    average_loss = MovingAverage(args.save_every)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model_encoder.to(device)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, [2, 4, 6, 8, 10, 12, 14, 16], 0.8)
    scheduler_encoder = optim.lr_scheduler.MultiStepLR(optimizer_encoder, [2, 4, 6, 8, 10, 12, 14, 16], 0.8)
    if args.restart:
        start_epoch = 0
        global_step = 0
        best_loss = np.inf
        print('=> no checkpoint file to be loaded.')
    else:
        try:
            checkpoint = load_checkpoint(checkpoint_dir, device == 'cuda', 'latest')
            start_epoch = checkpoint['epoch']
            global_step = checkpoint['global_iter']
            best_loss = checkpoint['best_loss']
            state_dict = checkpoint['state_dict']
            state_dict_encoder = checkpoint['state_dict_encoder']
            # new_state_dict = OrderedDict()
            # for k, v in state_dict.items():
            #     name = "model."+ k  # remove `module.`
            #     new_state_dict[name] = v
            model.load_state_dict(state_dict)
            model_encoder.load_state_dict(state_dict_encoder)
            optimizer.load_state_dict(checkpoint['optimizer'])
            print('=> loaded checkpoint (epoch {}, global_step {})'.format(start_epoch, global_step))
        except:
            start_epoch = 0
            global_step = 0
            best_loss = np.inf
            print('=> no checkpoint file to be loaded.')
    eps = 1e-4
    psnrs = []
    for epoch in range(start_epoch, args.epoch):
        for step, (noise, gt) in enumerate(data_loader):
            noise = noise.to(device)
            gt = gt.to(device)
            pred, pred_down = model(noise)
            # print(pred.size())
            encode_down = model_encoder(noise)
            loss = loss_func(pred, gt)
            # print(encode_down.size())
            # print(pred_down.size())
            # bs = gt.size()[0]
            # diff = noise - gt
            # loss = torch.sqrt((diff * diff) + (eps * eps))
            # loss = loss.view(bs,-1)
            # loss = adaptive.lossfun(loss)
            # loss = torch.mean(loss)
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()

            loss_encoder = loss_func(encode_down, pred_down)
            optimizer_encoder.zero_grad()
            loss_encoder.backward()
            optimizer_encoder.step()
            average_loss.update(loss)
            psnrs.append(calculate_psnr(pred, gt))
            if global_step % args.save_every == 0:
                # print(len(average_loss._cache))
                if average_loss.get_value() < best_loss:
                    is_best = True
                    best_loss = average_loss.get_value()
                else:
                    is_best = False

                save_dict = {
                    'epoch': epoch,
                    'global_iter': global_step,
                    'state_dict': model.state_dict(),
                    'state_dict_encoder': model_encoder.state_dict(),
                    'best_loss': best_loss,
                    'optimizer': optimizer.state_dict(),
                }
                save_checkpoint(save_dict, is_best, checkpoint_dir, global_step)
            if global_step % args.loss_every == 0:
                # print(global_step, "PSNR  : ", calculate_psnr(pred, gt))
                print(global_step, "PSNR  : ", calculate_psnr(pred, gt))
                print(average_loss.get_value())
            global_step += 1
        print('Epoch {} is finished.'.format(epoch))
        scheduler.step()
        scheduler_encoder.step()

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--noise_dir', '-n', default='/home/dell/Downloads/noise', help='path to noise folder image')
    parser.add_argument('--gt_dir', '-g', default='/home/dell/Downloads/gt', help='path to gt folder image')
    parser.add_argument('--image_size', '-sz', default=128, type=int, help='size of image')
    parser.add_argument('--batch_size', '-bs', default=2, type=int, help='batch size')
    parser.add_argument('--epoch', '-e', default=1000, type=int, help='batch size')
    parser.add_argument('--save_every', '-se', default=2, type=int, help='save_every')
    parser.add_argument('--loss_every', '-le', default=1, type=int, help='loss_every')
    parser.add_argument('--lr', default=1e-5, type=float, help='learning rate')
    parser.add_argument('--restart', '-r', action='store_true',
                        help='Whether to remove all old files and restart the training process')
    parser.add_argument('--num_workers', '-nw', default=4, type=int, help='number of workers in data loader')
    parser.add_argument('--cuda', '-c', action='store_true', help='whether to train on the GPU')
    parser.add_argument('--model_type','-m' ,default="KPN", help='type of model : KPN, MIR')
    parser.add_argument('--data_type', '-d', default="rgb", help='type of model : rgb, raw')
    parser.add_argument('--n_colors', '-nc', default=3,type=int, help='number of color dim')
    parser.add_argument('--out_channels', '-oc', default=3,type=int, help='number of out_channels')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='checkpoints',
                        help='the checkpoint to eval')

    args = parser.parse_args()
    #
    train(args)