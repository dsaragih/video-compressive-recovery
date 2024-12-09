import numpy as np 
import scipy.io as scio 
from torch.utils.data import Dataset 
import os 
import os.path as osp 
from .builder import DATASETS

@DATASETS.register_module
class GraySimDavis(Dataset):
    def __init__(self,data_root,*args,**kwargs):
        self.data_root = data_root
        self.data_name_list = os.listdir(data_root)
        self.mask = kwargs["mask"]
        # self.mask = mask
        self.frames,self.height,self.width = self.mask.shape

    def __getitem__(self,index):
        # Note how we're not accessing the meas or mask attributes here.
        # Instead we form the meas using the given mask from kwargs.
        pic = scio.loadmat(osp.join(self.data_root,self.data_name_list[index]))
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
        # pic = pic / 255
        # 256 x 256 x 32
        pic = pic[0:self.height,0:self.width,:]
        # print(f"Data name: {self.data_name_list[index]}")
        # print(f"Pic shape: {pic.shape}")
        # print(f"Frames: {self.frames}")
        # raise ValueError(f"Data name: {self.data_name_list[index]}, Pic shape: {pic.shape}, Frames: {self.frames}")
        # e.g. 2 x 16 x 256 x 256
        if pic.shape[2] // self.frames == 0:
            return np.zeros([self.frames, self.height, self.width]), np.zeros([1, self.frames, self.height, self.width])
        pic_gt = np.zeros([pic.shape[2] // self.frames, self.frames, self.height, self.width])
        # print(f"Pic shape: {pic.shape}")
        # print(f"Pic_gt shape: {pic_gt.shape}")
        for jj in range(pic_gt.shape[0]*self.frames):
            if jj % self.frames == 0:
                meas_t = np.zeros([self.height, self.width])
                n = 0
            pic_t = pic[:, :, jj]
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
        return meas,pic_gt
    def __len__(self,):
        return len(self.data_name_list)