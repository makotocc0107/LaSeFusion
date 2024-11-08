import argparse
import os
import random
import logging

import torch
import torch.nn.functional as F
import torchvision
from torch import optim
import numpy as np
from torch.utils.data import DataLoader
from dataloader import MSRS_data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from loss import *
from models.network_remake import LaSeFusion
from models.utlis import clamp, RGB2YCrCb, YCrCb2RGB

def init_seeds(seed=0):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)

def setup_logger(log_path):
    """Setup logger to save console output to a file."""
    log_dir = os.path.dirname(log_path)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)  # 创建保存日志文件的文件夹

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LaSeFusion")
    parser.add_argument('--dataset_path', metavar='DIR', default='datasets/msrs_train',
                        help='path to dataset (default: imagenet)')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fusion_train',
                        choices=['fusion_train', 'fusion_test'])
    parser.add_argument('--save_path', default='train_model')  # 模型存储路径
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--epochs', default=30, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                             'batch size of all GPUs on the current node when '
                             'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.0001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--image_size', default=64, type=int,
                        metavar='N', help='image size of input')
    parser.add_argument('--loss_weight', default='[3, 7, 10]', type=str,
                        metavar='N', help='loss weight')
    parser.add_argument('--cls_pretrained', default='pretrained/best_cls.pth',
                        help='use cls pre-trained model')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use GPU or not.')

    args = parser.parse_args()
    setup_logger(os.path.join(args.save_path, 'train_log.txt'))

    init_seeds(args.seed)

    train_datasets = MSRS_data(args.dataset_path)
    train_loader = DataLoader(train_datasets, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                              pin_memory=True)

    writer = SummaryWriter('./logs_train')

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.arch == "fusion_train":
        model = LaSeFusion()
        if torch.cuda.is_available():
            model = model.cuda()

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        best_loss = float('inf')  # Initialize best loss to a very high value

        for epoch in range(args.start_epoch, args.epochs):
            if epoch < args.epochs // 2:
                lr = args.lr
            else:
                lr = args.lr * (args.epochs - epoch) / (args.epochs - args.epochs // 2)

            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            model.train()
            epoch_loss = 0
            train_tqdm = tqdm(train_loader, total=len(train_loader))
            for vis_y_image, _, _, inf_image, _, _ in train_tqdm:
                vis_y_image = vis_y_image.cuda()
                inf_image = inf_image.cuda()
                optimizer.zero_grad()
                fused_image, vis_y_image_enhanced = model(vis_y_image, inf_image)

                final_fused_image = clamp(fused_image)
                loss_aux = F.l1_loss(fused_image, torch.max(vis_y_image, inf_image))
                gradient_loss = F.l1_loss(gradient(fused_image), torch.max(gradient(vis_y_image), gradient(inf_image)))
                loss_rec_y = YChannelEnhancementLoss()
                loss_rec = loss_rec_y(vis_y_image_enhanced, vis_y_image)

                t1, t2, t3 = eval(args.loss_weight)
                loss = t1 * loss_aux + t2 * gradient_loss + t3 * loss_rec

                train_tqdm.set_postfix(epoch=epoch, loss_aux=t1 * loss_aux.item(),
                                       loss_gradient=t2 * gradient_loss.item(),
                                       loss_rec=loss_rec.item(), loss_total=loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_loss += loss.item()

                writer.add_scalar("train_loss", loss.item(), epoch)

            epoch_loss /= len(train_loader)
            logging.info(f"Epoch {epoch}/{args.epochs} - Avg Loss: {epoch_loss:.4f}")

            # Save model at each epoch
            torch.save(model.state_dict(), f'{args.save_path}/model_epoch_{epoch}.pth')

            # Save the best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), f'{args.save_path}/best_model.pth')
                logging.info("Best model saved with loss: {:.4f}".format(best_loss))

            print("\n 模型保存完毕")
