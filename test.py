# 开发时间：2024/6/27 13:20
import argparse
import os
import random

import torch
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
from dataloader import MSRS_data
from tqdm import tqdm

from models.network_remake import LaSeFusion
from models.utlis import clamp, RGB2YCrCb, YCrCb2RGB

def init_seeds(seed=0):
    # Initialize random number generator (RNG) seeds https://pytorch.org/docs/stable/notes/randomness.html
    # cudnn seed 0 settings are slower and more reproducible, else faster and less reproducible
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if args.cuda:
        torch.cuda.manual_seed(seed)
    cudnn.benchmark, cudnn.deterministic = (False, True) if seed == 0 else (True, False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="LaSeFusion")
    parser.add_argument('--dataset_path', metavar='DIR', default='test_data/msrs_test',
                        help='path to dataset (default: imagenet)')  # 测试数据存放位置
    parser.add_argument('-a', '--arch', metavar='ARCH', default='fusion_test',
                        choices=['fusion_train', 'fusion_test'])
    parser.add_argument('--save_path', default='results/fusion')  # 融合结果存放位置
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('--train_models', default='train_model/model_epoch_0.pth',
                        help='use cls pre-trained model')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--cuda', default=True, type=bool,
                        help='use GPU or not.')

    args = parser.parse_args()

    init_seeds(args.seed)

    test_data = MSRS_data(args.dataset_path)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.arch == 'fusion_test':
        model = LaSeFusion()
        model = model.cuda()
        model.load_state_dict(torch.load(args.train_models))

        model.eval()
        test_tqdm = tqdm(test_loader, total=len(test_loader))
        with torch.no_grad():
            for vis_y_image, vis_cb_image, vis_cr_image, inf_image, name, _ in test_tqdm:
                vis_y_image = vis_y_image.cuda()
                vis_cb_image = vis_cb_image.cuda()
                vis_cr_image = vis_cr_image.cuda()
                inf_image = inf_image.cuda()

                fused_image = model(vis_y_image, inf_image)
                fused_image = clamp(fused_image)

                rgb_fused_image = YCrCb2RGB(fused_image[0], vis_cb_image[0], vis_cr_image[0])
                rgb_fused_image = transforms.ToPILImage()(rgb_fused_image)
                rgb_fused_image.save(f'{args.save_path}/{name[0]}')
