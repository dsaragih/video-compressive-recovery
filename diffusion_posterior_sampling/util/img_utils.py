import numpy as np
import torch
import scipy
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from motionblur.motionblur import Kernel
from .fastmri_utils import fft2c_new, ifft2c_new


"""
Helper functions for new types of inverse problems
"""

def fft2(x):
  """ FFT with shifting DC to the center of the image"""
  return torch.fft.fftshift(torch.fft.fft2(x), dim=[-1, -2])


def ifft2(x):
  """ IFFT with shifting DC to the corner of the image prior to transform"""
  return torch.fft.ifft2(torch.fft.ifftshift(x, dim=[-1, -2]))


def fft2_m(x):
  """ FFT for multi-coil """
  if not torch.is_complex(x):
      x = x.type(torch.complex64)
  return torch.view_as_complex(fft2c_new(torch.view_as_real(x)))


def ifft2_m(x):
  """ IFFT for multi-coil """
  if not torch.is_complex(x):
      x = x.type(torch.complex64)
  return torch.view_as_complex(ifft2c_new(torch.view_as_real(x)))


def clear(x):
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(x)


def clear_color(x):
    if torch.is_complex(x):
        x = torch.abs(x)
    x = x.detach().cpu().squeeze().numpy()
    return normalize_np(np.transpose(x, (1, 2, 0)))


def normalize_np(img):
    """ Normalize img in arbitrary range to [0, 1] """
    img -= np.min(img)
    img /= np.max(img) + 1e-8
    return img

def _format_fname(fname: str, i: int) -> str:
    """
    fname: /dir/0000x.png -> /dir/0000x_0000i.png
    """
    # Split the fname and get the last part
    parts = fname.split('/')
    last_part = parts[-1]
    # Split the last part and get the first part
    last_parts = last_part.split('.')
    first_part = last_parts[0]
    # Append the index to the first part
    first_part = first_part + '_' + str(i).zfill(5)

    # Join the parts and return
    parts[-1] = first_part + '.' + last_parts[1]
    return '/'.join(parts)


def visualize(path, x):
    # If instance of torch.Tensor
    if torch.is_tensor(x):
        # Check if batch or single image
        # If batch, loop through and save each image
        if x.shape[0] > 1:
            for i in range(x.shape[0]):
                plt.imsave(_format_fname(path, i), clear_color(x[i]))
        else:
            plt.imsave(path + '.png', clear_color(x))
    

def prepare_im(load_dir, image_size, device):
    ref_img = torch.from_numpy(normalize_np(plt.imread(load_dir)[:, :, :3].astype(np.float32))).to(device)
    ref_img = ref_img.permute(2, 0, 1)
    ref_img = ref_img.view(1, 3, image_size, image_size)
    ref_img = ref_img * 2 - 1
    return ref_img


def fold_unfold(img_t, kernel, stride):
    img_shape = img_t.shape
    B, C, H, W = img_shape
    print("\n----- input shape: ", img_shape)

    patches = img_t.unfold(3, kernel, stride).unfold(2, kernel, stride).permute(0, 1, 2, 3, 5, 4)

    print("\n----- patches shape:", patches.shape)
    # reshape output to match F.fold input
    patches = patches.contiguous().view(B, C, -1, kernel*kernel)
    print("\n", patches.shape) # [B, C, nb_patches_all, kernel_size*kernel_size]
    patches = patches.permute(0, 1, 3, 2)
    print("\n", patches.shape) # [B, C, kernel_size*kernel_size, nb_patches_all]
    patches = patches.contiguous().view(B, C*kernel*kernel, -1)
    print("\n", patches.shape) # [B, C*prod(kernel_size), L] as expected by Fold

    output = F.fold(patches, output_size=(H, W),
                    kernel_size=kernel, stride=stride)
    # mask that mimics the original folding:
    recovery_mask = F.fold(torch.ones_like(patches), output_size=(
        H, W), kernel_size=kernel, stride=stride)
    output = output/recovery_mask

    return patches, output


def reshape_patch(x, crop_size=128, dim_size=3):
    x = x.transpose(0, 2).squeeze()  # [9, 3*(128**2)]
    x = x.view(dim_size**2, 3, crop_size, crop_size)
    return x

def reshape_patch_back(x, crop_size=128, dim_size=3):
    x = x.view(dim_size**2, 3*(crop_size**2)).unsqueeze(dim=-1)
    x = x.transpose(0, 2)
    return x


def reshuffle_mosaic2vid(image, K, bucket=0):
    """
    Vectorized conversion of a mosaic image into K^2 low-resolution frames for a batch of images.
    
    Args:
    - image (numpy array): Input mosaic image of shape (batch, ch, H, W).
    - K (int): Size of each KxK tile.
    
    Returns:
    - frames (numpy array): Reshuffled frames of shape (batch, ch, K^2, H//K, W//K).
    """
    
    batch, ch, H, W = image.shape
    
    # Compute the padding required to make H and W divisible by K
    pad_h = (K - H % K) % K
    pad_w = (K - W % K) % K
    
    # Pad the image with zeros along height and width dimensions
    padded_image = np.pad(image, ((0, 0), (0, 0), (0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    
    # Get the new padded dimensions
    new_H, new_W = padded_image.shape[2], padded_image.shape[3]
    
    # Reshape the image to extract KxK tiles
    padded_image = padded_image[..., ::-1, :]  # Flip the image vertically along the height axis
    
    # Reshape the image to extract KxK tiles:
    # Shape: (batch, ch, H//K, K, W//K, K)
    reshaped_image = padded_image.reshape(batch, ch, new_H // K, K, new_W // K, K)
    
    # Transpose to get the KxK tile dimensions aligned for reshuffling
    # Now shape is: (batch, ch, H//K, W//K, K, K)
    transposed_image = reshaped_image.transpose(0, 1, 2, 4, 3, 5)
    
    # Reshape to combine the KxK tiles into K^2 separate frames
    # Shape: (batch, ch, H//K, W//K, K^2)
    flattened_image = transposed_image.reshape(batch, ch, new_H // K, new_W // K, K * K)
    
    # Transpose to get the final frame structure
    # Shape: (batch, ch, K^2, H//K, W//K)
    frames = flattened_image.transpose(0, 1, 4, 2, 3)
    
    # Optionally reverse the frames or the height dimension depending on the bucket value
    # frames = frames[:, :, ::-1, ::-1, :] if bucket == 0 else frames[:, :, :, ::-1, :]
    if bucket == 0:
        # Flip along axis 2 and 3
        frames = np.flip(frames, axis=(3))
    elif bucket == 1:
        # Flip along axis 3
        frames = np.flip(frames, axis=3)
    
    return frames.copy()

class Unfolder:
    def __init__(self, img_size=256, crop_size=128, stride=64):
        self.img_size = img_size
        self.crop_size = crop_size
        self.stride = stride

        self.unfold = nn.Unfold(crop_size, stride=stride)
        self.dim_size = (img_size - crop_size) // stride + 1

    def __call__(self, x):
        patch1D = self.unfold(x)
        patch2D = reshape_patch(patch1D, crop_size=self.crop_size, dim_size=self.dim_size)
        return patch2D


def center_crop(img, new_width=None, new_height=None):

    width = img.shape[1]
    height = img.shape[0]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))

    if len(img.shape) == 2:
        center_cropped_img = img[top:bottom, left:right]
    else:
        center_cropped_img = img[top:bottom, left:right, ...]

    return center_cropped_img

class Folder:
    def __init__(self, img_size=256, crop_size=128, stride=64):
        self.img_size = img_size
        self.crop_size = crop_size
        self.stride = stride

        self.fold = nn.Fold(img_size, crop_size, stride=stride)
        self.dim_size = (img_size - crop_size) // stride + 1

    def __call__(self, patch2D):
        patch1D = reshape_patch_back(patch2D, crop_size=self.crop_size, dim_size=self.dim_size)
        return self.fold(patch1D)


def random_sq_bbox(img, mask_shape, image_size=256, margin=(16, 16)):
    """Generate a random sqaure mask for inpainting
    """
    B, C, H, W = img.shape
    h, w = mask_shape
    margin_height, margin_width = margin
    maxt = image_size - margin_height - h
    maxl = image_size - margin_width - w

    # bb
    t = np.random.randint(margin_height, maxt)
    l = np.random.randint(margin_width, maxl)

    # make mask
    mask = torch.ones([B, C, H, W], device=img.device)
    mask[..., t:t+h, l:l+w] = 0

    return mask, t, t+h, l, l+w


class mask_generator:
    def __init__(self, mask_type, mask_len_range=None, mask_prob_range=None,
                 image_size=256, margin=(16, 16)):
        """
        (mask_len_range): given in (min, max) tuple.
        Specifies the range of box size in each dimension
        (mask_prob_range): for the case of random masking,
        specify the probability of individual pixels being masked
        """
        assert mask_type in ['box', 'random', 'both', 'extreme']
        self.mask_type = mask_type
        self.mask_len_range = mask_len_range
        self.mask_prob_range = mask_prob_range
        self.image_size = image_size
        self.margin = margin

    def _retrieve_box(self, img):
        l, h = self.mask_len_range
        l, h = int(l), int(h)
        mask_h = np.random.randint(l, h)
        mask_w = np.random.randint(l, h)
        mask, t, tl, w, wh = random_sq_bbox(img,
                              mask_shape=(mask_h, mask_w),
                              image_size=self.image_size,
                              margin=self.margin)
        return mask, t, tl, w, wh

    def _retrieve_random(self, img):
        total = self.image_size ** 2
        # random pixel sampling
        l, h = self.mask_prob_range
        prob = np.random.uniform(l, h)
        mask_vec = torch.ones([1, self.image_size * self.image_size])
        samples = np.random.choice(self.image_size * self.image_size, int(total * prob), replace=False)
        mask_vec[:, samples] = 0
        mask_b = mask_vec.view(1, self.image_size, self.image_size)
        mask_b = mask_b.repeat(3, 1, 1)
        mask = torch.ones_like(img, device=img.device)
        mask[:, ...] = mask_b
        return mask

    def __call__(self, img):
        if self.mask_type == 'random':
            mask = self._retrieve_random(img)
            return mask
        elif self.mask_type == 'box':
            mask, t, th, w, wl = self._retrieve_box(img)
            return mask
        elif self.mask_type == 'extreme':
            mask, t, th, w, wl = self._retrieve_box(img)
            mask = 1. - mask
            return mask

def unnormalize(img, s=0.95):
    scaling = torch.quantile(img.abs(), s)
    return img / scaling


def normalize(img, s=0.95):
    scaling = torch.quantile(img.abs(), s)
    return img * scaling


def dynamic_thresholding(img, s=0.95):
    img = normalize(img, s=s)
    return torch.clip(img, -1., 1.)


def get_gaussian_kernel(kernel_size=31, std=0.5):
    n = np.zeros([kernel_size, kernel_size])
    n[kernel_size//2, kernel_size//2] = 1
    k = scipy.ndimage.gaussian_filter(n, sigma=std)
    k = k.astype(np.float32)
    return k


def init_kernel_torch(kernel, device="cuda:0"):
    h, w = kernel.shape
    kernel = Variable(torch.from_numpy(kernel).to(device), requires_grad=True)
    kernel = kernel.view(1, 1, h, w)
    kernel = kernel.repeat(1, 3, 1, 1)
    return kernel

class CodeGenerator:
    # Generates a mask of size (subframes, 256, 256)
    # Each sub-mask is a partition of tiles of size sqrt(subframes) x sqrt(subframes)
    # For simple code, with subframes = 4
    """
    mask_tile[0] = [[1, 0], [0, 0]]
    mask_tile[1] = [[0, 1], [0, 0]]
    mask_tile[2] = [[0, 0], [1, 0]]
    mask_tile[3] = [[0, 0], [0, 1]]
    """
    def __init__(self, subframes, code_type='simple', image_w=256, image_h=256):
        self.subframes = subframes
        self.code_type = code_type
        self.image_w = image_w
        self.image_h = image_h

    def get_pattern(self, K):
        # Initialize an empty list to hold the patterns
        patterns = []
        
        # Generate K^2 patterns
        for i in range(K):
            for j in range(K):
                # Create a K x K matrix filled with zeros
                pattern = np.zeros((K, K), dtype=int)
                # Set the (i, j)-th element to 1
                pattern[i, j] = 1
                # Flip pattern along the horizontal axis
                pattern = np.flip(pattern, axis=0)
                # Append this pattern to the list
                patterns.append(pattern)
        
        # Convert the list of patterns to a 3D numpy array of shape (K^2, K, K)
        patterns_array = np.array(patterns)
        
        return patterns_array

    def get_mask(self):
        if self.code_type == 'simple':
            tile_size = int(np.sqrt(self.subframes))
            img_height = self.image_h
            img_width = self.image_w

            patterns = self.get_pattern(tile_size)

            mask = np.zeros((self.subframes, img_height, img_width), dtype=np.uint8)
            tile_multiplier_height = img_height // tile_size + 1 # we cut off the extra pixels later
            tile_multiplier_width = img_width // tile_size + 1

            for i in range(self.subframes):
                pattern = patterns[i]
                mask[i] = np.tile(pattern, (tile_multiplier_height, tile_multiplier_width))[:img_height, :img_width]
            
            mask = mask.reshape(self.subframes * img_height, img_width)
            mask = torch.from_numpy(mask).float()
            mask = mask.unsqueeze(0).unsqueeze(0)
        else:
            raise ValueError("Invalid code type")
        # Flip height dimension [batch, 1, 256, 256]
        # return torch.flip(mask, [2])
        return mask


class Blurkernel(nn.Module):
    def __init__(self, blur_type='gaussian', kernel_size=31, std=3.0, device=None):
        super().__init__()
        self.blur_type = blur_type
        self.kernel_size = kernel_size
        self.std = std
        self.device = device
        self.seq = nn.Sequential(
            nn.ReflectionPad2d(self.kernel_size//2),
            nn.Conv2d(3, 3, self.kernel_size, stride=1, padding=0, bias=False, groups=3)
        )

        self.weights_init()

    def forward(self, x):
        return self.seq(x)

    def weights_init(self):
        if self.blur_type == "gaussian":
            n = np.zeros((self.kernel_size, self.kernel_size))
            n[self.kernel_size // 2,self.kernel_size // 2] = 1
            k = scipy.ndimage.gaussian_filter(n, sigma=self.std)
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)
        elif self.blur_type == "motion":
            k = Kernel(size=(self.kernel_size, self.kernel_size), intensity=self.std).kernelMatrix
            k = torch.from_numpy(k)
            self.k = k
            for name, f in self.named_parameters():
                f.data.copy_(k)

    def update_weights(self, k):
        if not torch.is_tensor(k):
            k = torch.from_numpy(k).to(self.device)
        for name, f in self.named_parameters():
            f.data.copy_(k)

    def get_kernel(self):
        return self.k


class exact_posterior():
    def __init__(self, betas, sigma_0, label_dim, input_dim):
        self.betas = betas
        self.sigma_0 = sigma_0
        self.label_dim = label_dim
        self.input_dim = input_dim

    def py_given_x0(self, x0, y, A, verbose=False):
        norm_const = 1/((2 * np.pi)**self.input_dim * self.sigma_0**2)
        exp_in = -1/(2 * self.sigma_0**2) * torch.linalg.norm(y - A(x0))**2
        if not verbose:
            return norm_const * torch.exp(exp_in)
        else:
            return norm_const * torch.exp(exp_in), norm_const, exp_in

    def pxt_given_x0(self, x0, xt, t, verbose=False):
        beta_t = self.betas[t]
        norm_const = 1/((2 * np.pi)**self.label_dim * beta_t)
        exp_in = -1/(2 * beta_t) * torch.linalg.norm(xt - np.sqrt(1 - beta_t)*x0)**2
        if not verbose:
            return norm_const * torch.exp(exp_in)
        else:
            return norm_const * torch.exp(exp_in), norm_const, exp_in

    def prod_logsumexp(self, x0, xt, y, A, t):
        py_given_x0_density, pyx0_nc, pyx0_ei = self.py_given_x0(x0, y, A, verbose=True)
        pxt_given_x0_density, pxtx0_nc, pxtx0_ei = self.pxt_given_x0(x0, xt, t, verbose=True)
        summand = (pyx0_nc * pxtx0_nc) * torch.exp(-pxtx0_ei - pxtx0_ei)
        return torch.logsumexp(summand, dim=0)



def map2tensor(gray_map):
    """Move gray maps to GPU, no normalization is done"""
    return torch.FloatTensor(gray_map).unsqueeze(0).unsqueeze(0).cuda()


def create_penalty_mask(k_size, penalty_scale):
    """Generate a mask of weights penalizing values close to the boundaries"""
    center_size = k_size // 2 + k_size % 2
    mask = create_gaussian(size=k_size, sigma1=k_size, is_tensor=False)
    mask = 1 - mask / np.max(mask)
    margin = (k_size - center_size) // 2 - 1
    mask[margin:-margin, margin:-margin] = 0
    return penalty_scale * mask


def create_gaussian(size, sigma1, sigma2=-1, is_tensor=False):
    """Return a Gaussian"""
    func1 = [np.exp(-z ** 2 / (2 * sigma1 ** 2)) / np.sqrt(2 * np.pi * sigma1 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    func2 = func1 if sigma2 == -1 else [np.exp(-z ** 2 / (2 * sigma2 ** 2)) / np.sqrt(2 * np.pi * sigma2 ** 2) for z in range(-size // 2 + 1, size // 2 + 1)]
    return torch.FloatTensor(np.outer(func1, func2)).cuda() if is_tensor else np.outer(func1, func2)


def total_variation_loss(img, weight):
    tv_h = ((img[:, :, 1:, :] - img[:, :, :-1, :]).pow(2)).mean()
    tv_w = ((img[:, :, :, 1:] - img[:, :, :, :-1]).pow(2)).mean()
    return weight * (tv_h + tv_w)


if __name__ == '__main__':
    import numpy as np
    from torch import nn
    import matplotlib.pyplot as plt
    device = 'cuda:0'
    load_path = '/media/harry/tomo/FFHQ/256/test/00000.png'
    img = torch.tensor(plt.imread(load_path)[:, :, :3])  #rgb
    img = torch.permute(img, (2, 0, 1)).view(1, 3, 256, 256).to(device)

    mask_len_range = (32, 128)
    mask_prob_range = (0.3, 0.7)
    image_size = 256
    # mask
    mask_gen = mask_generator(
        mask_len_range=mask_len_range,
        mask_prob_range=mask_prob_range,
        image_size=image_size
    )
    mask = mask_gen(img)

    mask = np.transpose(mask.squeeze().cpu().detach().numpy(), (1, 2, 0))

    plt.imshow(mask)
    plt.show()
