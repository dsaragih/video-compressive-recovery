'''
------------------------------------------
DEFINE DATALOADER TO FETCH VIDEO SEQUENCES
------------------------------------------
removed recurrence in train data
added return of patches
added hdf5 file access

'''
import scipy.io as scio 
import os.path as osp 
import torch
from torch.utils import data
import glob
import os
import numpy as np
import scipy.misc
from PIL import Image
import h5py
import cv2
from utils import Compose
# import time


class Dataset_load(data.Dataset):
    
    def __init__(self, filepath, dataset, num_samples='all'):
        'Initialization'

        f = h5py.File(filepath, 'r')
        print('\nReading data from %s'%(filepath))
        print('Found', list(f.keys()), '...Reading from', dataset)
        
        if num_samples == 'all':
            self.dset = f[dataset]
        else:
            self.dset = f[dataset][:num_samples]


    def __len__(self):
        'Denotes the total number of samples'

        return self.dset.shape[0]


    def __getitem__(self, index):
        'Generates one sample of data'

        vid = torch.FloatTensor(self.dset[index, ...]) / 255.
        return vid
    

class DavisData(data.Dataset):
    def __init__(self,data_root,*args,**kwargs):

        self.data_dir= data_root
        self.data_list = os.listdir(data_root)
        self.img_files = []
        self.mask = kwargs["mask"]
        self.pipeline = Compose(kwargs["pipeline"])

        self.ratio,self.resize_w,self.resize_h = self.mask.shape

        for image_dir in os.listdir(data_root):

            train_data_path = osp.join(data_root,image_dir)
            data_path = os.listdir(train_data_path)
            data_path.sort()
            for sub_index in range(len(data_path)-self.ratio):
                sub_data_path = data_path[sub_index:]
                image_name_list = []
                count = 0
                for image_name in sub_data_path:
                    image_name_list.append(osp.join(train_data_path,image_name))
                    if (count+1)%self.ratio==0:
                        self.img_files.append(image_name_list)
                        image_name_list = []
                    count += 1
        

    def refine(self, imgs):
        assert isinstance(imgs,list), "imgs must be list"
        gt = []

        for i,img in enumerate(imgs):
            Y = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)[:,:,0]
            Y = Y.astype(np.float32)/255
            gt.append(Y)

        return np.array(gt)
    
    def __getitem__(self,index):
        imgs = []
        for i,image_path in enumerate(self.img_files[index]):
            img = cv2.imread(image_path)
            imgs.append(img)
        # print(f"Image shape: {imgs[0].shape}")
        imgs = self.pipeline(imgs)
        return self.refine(imgs)
    
    def __len__(self,):
        return len(self.img_files)

class SixGraySimData(data.Dataset):
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
    
class GraySimDavis(data.Dataset):
    def __init__(self,data_root,*args,**kwargs):
        self.data_root = data_root
        self.data_name_list = os.listdir(data_root)
        self.mask = kwargs["mask"]
        # self.mask = mask
        self.frames,self.height,self.width = self.mask.shape

    def __getitem__(self,index):
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
        for jj in range(pic_gt.shape[0]*self.frames):
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