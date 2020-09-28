from torch.utils.data import Dataset
import os
import glob
import cv2
import random
import torch


class Div2K(Dataset):

    def __init__(self, data_root, scale, patch_size):

        self.data_dir = data_root
        self.scale = scale
        self.patch_size = patch_size

        self.hr_folder = os.path.join(self.data_dir, 'DIV2K_train_HR')
        self.lr_folder = os.path.join(self.data_dir, 'DIV2K_train_LR_bicubic', 'X' + str(self.scale))

        self.extension = 'png'
        self.hr_image_names = glob.glob(self.hr_folder + '/*.png')

    def __len__(self):

        return len(self.hr_image_names)

    def __getitem__(self, item):

        hr_img_name = self.hr_image_names[item]
        hr_img = cv2.imread(hr_img_name)
        lr_img_name = os.path.join(self.lr_folder,
                                   os.path.splitext(os.path.basename(hr_img_name))[0] + 'x' + str(self.scale) + '.png')
        lr_img = cv2.imread(lr_img_name)

        hr_patch, lr_patch = self.extract_patch(hr_img, lr_img)
        hr_patch_tensor, lr_patch_tensor = torch.from_numpy(hr_patch), torch.from_numpy(lr_patch)

        return hr_patch_tensor.permute(2, 0, 1).type(torch.float32), lr_patch_tensor.permute(2, 0, 1).type(torch.float32)

    def extract_patch(self, hr_img, lr_img):

        hr_patch_size = self.scale * self.patch_size
        h_lr, w_lr = lr_img.shape[:2]
        x_lr = random.randrange(0, w_lr - self.patch_size + 1)
        y_lr = random.randrange(0, h_lr - self.patch_size + 1)
        x_hr, y_hr = x_lr * self.scale, y_lr * self.scale

        return hr_img[y_hr:y_hr + hr_patch_size, x_hr:x_hr + hr_patch_size, :], lr_img[y_lr:y_lr + self.patch_size, x_lr:x_lr + self.patch_size, :]