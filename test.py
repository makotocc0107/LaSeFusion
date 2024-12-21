# 开发时间：2024/6/27 13:20
import argparse
import os
import random
import platform

import torch
from sympy.physics.units import boltzmann
from torchvision import transforms
import numpy as np
from torch.utils.data import DataLoader
from dataloader import MSRS_data
from tqdm import tqdm

from models.network_remake import LaSeFusion
from models.utlis import clamp, RGB2YCrCb, YCrCb2RGB

def select_device():
    system = platform.system()
    if system == 'Windows' or system == 'Linux':
        if torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"
    elif system == 'Darwin':
        if torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"
    else:
        return "cpu"

def init_seeds(seed=0):
    import torch.backends.cudnn as cudnn
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if platform.system() == "Windows" or platform.system() == "Linux":
        if torch.cuda.is_available():
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
    parser.add_argument('--train_models', default='train_model/best_model.pth',
                        help='use cls pre-trained model')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--usd_gpu', default=True, type=bool,
                        help='use GPU for training. ')
    args = parser.parse_args()

    init_seeds(args.seed)

    test_data = MSRS_data(args.dataset_path)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=args.workers, pin_memory=True)

    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)

    if args.arch == 'fusion_test':
        model = LaSeFusion()
        selected_device = select_device()
        if args.usd_gpu:
            device = torch.device(selected_device)
            print(f"Selected Device: {selected_device}")
        else:
            device = torch.device('cpu')
            print(f"Selected Device: cpu")

        # 加载训练好的模型权重
        if os.path.exists(args.train_models):
            if selected_device == 'mps':
                checkpoint = torch.load(args.train_models, map_location='cpu')  # 指定映射到当前设备
                model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
            else:
                checkpoint = torch.load(args.train_models, map_location=selected_device)
                model.load_state_dict(checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint)
            model.to(device)
            print(f"Loaded model weights from {args.train_models}")
        else:
            print(f"Model weights not found at {args.train_models}")
            exit()

        model.eval()
        test_tqdm = tqdm(test_loader, total=len(test_loader))
        with torch.no_grad():
            for vis_y_image, vis_cb_image, vis_cr_image, inf_image, name, _ in test_tqdm:
                vis_y_image = vis_y_image.to(device)
                vis_cb_image = vis_cb_image.to(device)
                vis_cr_image = vis_cr_image.to(device)
                inf_image = inf_image.to(device)

                fused_image = model(vis_y_image, inf_image)
                # fused_image = clamp(fused_image)

                rgb_fused_image = YCrCb2RGB(fused_image[0], vis_cb_image[0], vis_cr_image[0])
                rgb_fused_image = transforms.ToPILImage()(rgb_fused_image)
                rgb_fused_image.save(f'{args.save_path}/{name[0]}')
