import torch.nn as nn
import torch
import numpy as np
from torch.autograd import Variable
import torch.autograd as autograd
import argparse, json, glob, os, sys
Tensor = torch.cuda.FloatTensor 

from gan_helpers.spectral_norm import SpectralNorm


def split_into_patches(x, patch_size = 64):
    patches = torch.empty([1,x.shape[1],patch_size,patch_size], device = x.device)
    for xx in range(0,x.shape[-2]//patch_size):
        for yy in range(0,x.shape[-1]//patch_size):
            patches = torch.cat([patches, x[...,xx*patch_size:(xx+1)*patch_size, yy*patch_size:(yy+1)*patch_size]], 0)
    patches = patches[1:,...]
    return patches


def get_histogram(data, bin_edges=None, left_edge=0.0, right_edge=1.0, n_bins=1000):
    data_range = right_edge - left_edge
    bin_width = data_range / n_bins
    if bin_edges is None:
        bin_edges = np.arange(left_edge, right_edge + bin_width, bin_width)
    bin_centers = bin_edges[:-1] + (bin_width / 2.0)
    n = np.prod(data.shape)
    hist, _ = np.histogram(data, bin_edges)
    return hist / n, bin_centers


def cal_kld(p_data, q_data, left_edge=0.0, right_edge=1.0, n_bins=1000):
    """Returns forward, inverse, and symmetric KL divergence between two sets of data points p and q"""
    bw = 0.2 / 64
    bin_edges = np.concatenate(([-1000.0], np.arange(-0.1, 0.1 + 1e-9, bw), [1000.0]), axis=0)
    bin_edges = None
    p, _ = get_histogram(p_data, bin_edges, left_edge, right_edge, n_bins)
    q, _ = get_histogram(q_data, bin_edges, left_edge, right_edge, n_bins)
    idx = (p > 0) & (q > 0)
    p = p[idx]
    q = q[idx]
    logp = np.log(p)
    logq = np.log(q)
    kl_fwd = np.sum(p * (logp - logq))
    kl_inv = np.sum(q * (logq - logp))
    kl_sym = (kl_fwd + kl_inv) / 2.0
    return kl_sym


def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    alpha = torch.tensor(np.random.random((real_samples.size(0), 1, 1, 1)), dtype = real_samples.dtype, device = real_samples.device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)[...,0]
    fake = Variable(Tensor(d_interpolates.shape[0], 1).fill_(1.0), requires_grad=False).view(-1)

    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


class NoiseGenerator(nn.Module):
    def __init__(self, device, data_dir, r=320, c=320, n=1):
        super(NoiseGenerator, self).__init__()

        self.device = device
        self.dtype = torch.float32

        fwc = np.load(os.path.join(data_dir, "fwc.npy"))
        self.fwc = torch.tensor(fwc.transpose(2, 0, 1), dtype=self.dtype, device=device).unsqueeze(0)

        self.g = torch.nn.Parameter(torch.ones((1, n, r, c), dtype=self.dtype, device=device), requires_grad=True)

        self.h = torch.nn.Parameter(torch.zeros((1, n, r, c), dtype=self.dtype, device=device), requires_grad=True)

        self.shot = torch.nn.Parameter(torch.tensor(0.4, dtype=self.dtype, device=device), requires_grad=True)

        self.read = torch.nn.Parameter(torch.full((1, n, r, c), 15, dtype=self.dtype, device=device), requires_grad=True)

        self.row = torch.nn.Parameter(torch.tensor(5, dtype=self.dtype, device=device), requires_grad=True)

        self.rowt = torch.nn.Parameter(torch.tensor(5, dtype=self.dtype, device=device), requires_grad=True)

        self.quant = torch.nn.Parameter(torch.tensor(5, dtype=self.dtype, device=device), requires_grad=True)

        self.dark = torch.nn.Parameter(torch.full((1, n, r, c), 0.03, dtype=self.dtype, device=device), requires_grad=True)

    def forward(self, x, t, add_noise):
        z = x * self.g + self.h
        z = torch.where(z < self.fwc, z, self.fwc)

        if add_noise:
            shot = torch.randn(z.shape, device=self.device, requires_grad=True) * self.shot * torch.sqrt(x) * self.g
            shot = torch.where(z < self.fwc, shot, 0)
            z += shot

            read = torch.randn(z.shape, device=self.device, requires_grad=True) * self.read
            z += read

            row = torch.randn((*z.shape[:3], 1), device=self.device, requires_grad=True) * self.row
            z += row

            rowt = torch.randn((1, *z.shape[1:3], 1), device=self.device, requires_grad=True) * self.rowt
            z += rowt

            quant = (torch.rand(z.shape, device=self.device, requires_grad=True) - 0.5) * self.quant
            z += quant

            dark = torch.randn(z.shape, device=self.device, requires_grad=True) * self.dark * torch.sqrt(t)
            z += dark
            
        z = torch.clip(z, 0, 2**12-1)

        return z


class Discriminator(nn.Module):
    def __init__(self, n=1):
        super(Discriminator, self).__init__()
        
        self.conv1 = SpectralNorm(nn.Conv2d(n, 64, 3, stride=1, padding=(1, 1)))
        self.conv2 = SpectralNorm(nn.Conv2d(64, 128, 4, stride=2, padding=(1, 1)))
        self.conv3 = SpectralNorm(nn.Conv2d(128, 256, 4, stride=2, padding=(1, 1)))
        self.conv4 = SpectralNorm(nn.Conv2d(256, 512, 4, stride=2, padding=(1, 1)))
        self.conv5 = SpectralNorm(nn.Conv2d(512, 512 * 2, 3, stride=2, padding=(1, 1)))
        self.classifier = nn.Sequential(nn.Sigmoid())
        self.fc = SpectralNorm(nn.Linear(1024 * 4 * 4, 1))

    def forward(self, x):
        leak = 0.1

        m = x
        m = nn.LeakyReLU(leak)(self.conv1(m))
        m = nn.LeakyReLU(leak)(self.conv2(m))
        m = nn.LeakyReLU(leak)(self.conv3(m))
        m = nn.LeakyReLU(leak)(self.conv4(m))
        m = nn.LeakyReLU(leak)(self.conv5(m))
        out = m.view(m.shape[0],-1)
        out = self.fc(out)
        out = self.classifier(out)

        return out
