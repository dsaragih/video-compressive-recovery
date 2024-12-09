import glob
import numpy as np
import torch
import os


class ToTensor(object):
    def __call__(self, sample):
        for key in sample:
            if key != "exp":
                if len(sample[key].shape) == 4:
                    sample[key] = torch.tensor(sample[key].transpose(0, 3, 1, 2), dtype=torch.float32)
                else:
                    print("bad shape")
                    exit()
            else:
                if len(sample[key].shape) == 1:
                    sample[key] = torch.tensor(sample[key][0], dtype=torch.float32)
                else:
                    print("bad shape")
                    exit()

        return sample


class GetSampleBatch(object):
    def __init__(self, img_dir, transform, bsize=16):
        self.exp_path = os.path.join(img_dir, "exp.npy")
        self.clean_path = os.path.join(img_dir, "clean.npy")
        self.noisy_paths = sorted(glob.glob(img_dir + "/noisy*.npy"))
        self.transform = transform
        self.bsize = bsize

    def __len__(self):
       return len(self.noisy_paths) // self.bsize

    def __getitem__(self, idx):
        exp = np.load(self.exp_path).astype(np.float32)

        clean = np.load(self.clean_path).astype(np.float32)

        noisy_imgs = np.empty((self.bsize, *clean.shape))
        for i in range(0, self.bsize):
            j = idx * self.bsize + i
            noisy_imgs[i] = np.load(self.noisy_paths[j]).astype(np.float32)

        clean_imgs = np.repeat(clean[None], self.bsize, axis=0)
        
        sample = {"noisy_imgs": noisy_imgs, "clean_imgs": clean_imgs, "exp": exp}

        if self.transform:
            sample = self.transform(sample)
        else:
            print("no transform")
            exit()

        return sample
