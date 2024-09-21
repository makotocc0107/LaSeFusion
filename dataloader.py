# 开发时间：2024/5/30 14:21
from torch.utils.data import Dataset
from PIL import Image
import os
from torchvision import transforms
from models.fusion_network import RGB2YCrCb


to_tensor = transforms.Compose([transforms.ToTensor()])


class MSRS_data(Dataset):
    def __init__(self, root_dir, transform=to_tensor):
        super().__init__()
        self.root_dir = root_dir
        dirname = os.listdir(root_dir)
        for sub_dir in dirname:
            temp_path = os.path.join(root_dir, sub_dir)
            if sub_dir == "Infrared":



                self.inf_path = temp_path
            elif sub_dir == "Visible":
                self.vis_path = temp_path

        self.namelist = os.listdir(self.inf_path)
        self.transform = transform

    def __getitem__(self, idx):
        img_name = self.namelist[idx]

        inf_image = Image.open(os.path.join(self.inf_path, img_name)).convert("L")
        vis_image = Image.open(os.path.join(self.vis_path, img_name))
        inf_image = self.transform(inf_image)
        vis_image = self.transform(vis_image)
        vis_y_image, vis_cb_image, vis_cr_image = RGB2YCrCb(vis_image)
        return vis_y_image, vis_cb_image, vis_cr_image, inf_image, img_name, vis_image

    def __len__(self):
        return len(self.namelist)


