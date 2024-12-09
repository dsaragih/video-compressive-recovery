from glob import glob
from PIL import Image
from typing import Callable, Optional
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
import torch
import os
import numpy as np
import scipy.io as scio

__DATASET__ = {}

def register_dataset(name: str):
    def wrapper(cls):
        if __DATASET__.get(name, None):
            raise NameError(f"Name {name} is already registered!")
        __DATASET__[name] = cls
        return cls
    return wrapper


def get_dataset(name: str, root: str, **kwargs):
    if __DATASET__.get(name, None) is None:
        raise NameError(f"Dataset {name} is not defined.")
    return __DATASET__[name](root=root, **kwargs)


def get_dataloader(dataset: VisionDataset,
                   batch_size: int, 
                   num_workers: int, 
                   train: bool):
    dataloader = DataLoader(dataset, 
                            batch_size, 
                            shuffle=train, 
                            num_workers=num_workers, 
                            drop_last=train)
    return dataloader


@register_dataset(name='ffhq')
class FFHQDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)

        self.fpaths = sorted(glob(root + '/**/*.png', recursive=True))
        assert len(self.fpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, index: int):
        fpath = self.fpaths[index]
        img = Image.open(fpath).convert('RGB')
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img

@register_dataset(name='sgs')
class SixGraySimData(VisionDataset):
    def __init__(self,data_root,*args,**kwargs):
        self.data_root = data_root
        self.data_name_list = os.listdir(data_root)
        self.mask = kwargs["mask"]
        # self.mask = mask
        self.frames,self.height,self.width = self.mask.shape

    def __getitem__(self,index):
        # Note how we're not accessing the meas or mask attributes here.
        # Instead we form the meas using the given mask from kwargs.
        # mask her eis 1 x 4 x 256 x 256
        pic = scio.loadmat(os.path.join(self.data_root,self.data_name_list[index]))
        if "orig" in pic:
            pic = pic['orig']
        elif "patch_save" in pic:
            pic = pic['patch_save']
        elif "p1" in pic:
            pic = pic['p1']
        elif "p2" in pic:
            pic = pic['p2']
        elif "p3" in pic:
            pic = pic['p3']
        pic = pic / 255
        # 256 x 256 x 32
        pic = pic[0:self.height,0:self.width,:]
        # -> 32 x 256 x 256
        pic = np.transpose(pic, [2, 0, 1])
        # e.g. 2 x 16 x 256 x 256
        if pic.shape[0] // self.frames == 0:
            return np.zeros([self.frames, self.height, self.width]), np.zeros([self.frames, self.height, self.width])
        pic_gt = np.zeros([pic.shape[0] // self.frames, self.frames, self.height, self.width])
        # print(f"Pic shape: {pic.shape}")
        # print(f"Pic_gt shape: {pic_gt.shape}")
        for jj in range(pic.shape[0]):
            if jj % self.frames == 0:
                meas_t = np.zeros([1, self.height, self.width])
                n = 0
            pic_t = pic[jj, :, :]
            mask_t = self.mask[n, :, :]

            pic_gt[jj // self.frames, n, :, :] = pic_t
            n += 1
            meas_t = meas_t + np.multiply(mask_t, pic_t)

            if jj == (self.frames-1):
                # First time.
                meas_t = np.expand_dims(meas_t, 0)
                meas = meas_t
            elif (jj + 1) % self.frames == 0 and jj != (self.frames-1):
                meas_t = np.expand_dims(meas_t, 0)
                meas = np.concatenate((meas, meas_t), axis=0)
        return meas / np.sum(self.mask, axis=0, keepdims=True), pic_gt
    def __len__(self,):
        return len(self.data_name_list)


@register_dataset(name='video')
class VideoDataset(VisionDataset):
    def __init__(self, root: str, transforms: Optional[Callable]=None):
        super().__init__(root, transforms)
        
        # Root contains folders like 00001, 00002, ...
        # consisting of x frames in png format
        self.dpaths = sorted(glob(root + '/*'))

        assert len(self.dpaths) > 0, "File list is empty. Check the root."

    def __len__(self):
        return len(self.dpaths)

    def __getitem__(self, index: int):
        # Load all frames in the folder
        dpath = self.dpaths[index]
        fpaths = sorted(glob(dpath + '/*.png'))
        
        # Return torch tensor of shape [b, c, t, h, w]
        # where t is the number of frames
        imgs = []
        for fpath in fpaths:
            img = Image.open(fpath).convert('RGB')
            if self.transforms is not None:
                img = self.transforms(img)
            imgs.append(img)
        video = torch.stack(imgs, dim=0)
        return torch.permute(video, (1, 0, 2, 3))
