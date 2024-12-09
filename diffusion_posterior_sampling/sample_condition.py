from functools import partial
import os
import argparse
import yaml
import time

import torch
import torchvision.transforms as transforms
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity
import numpy as np
# Einops
from einops import rearrange

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, mask_generator, reshuffle_mosaic2vid, visualize
from util.logger import get_logger


def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def microbatch_sample(x_start, y_n, sample_fn, measurement_cond_fn, microbatch_size, device, logger, subframes, out_path):
    """
    Perform microbatch sampling.
    
    Args:
    - x_start (torch.Tensor): Input tensor of shape [b, c, h, w].
    - y_n (torch.Tensor or list of torch.Tensor): Tensor or list of tensors with the same shape as x_start.
    - sample_fn (function): Function to process each microbatch. Signature: sample_fn(x_start, measurement, record, save_root).
    - measurement_cond_fn (function): Function to condition the sampling process. Signature: measurement_cond_fn(x_prev, x_t, x_0_hat, measurement, noisy_measurement).
    - microbatch_size (int): Size of each microbatch.
    - device (torch.device): The device (CPU/GPU) for computation.
    - out_path (str): Directory where the results are saved.

    Returns:
    - sample (torch.Tensor): Output tensor of shape [b, c, h, w] after processing all microbatches.
    """
    
    # Split x_start into microbatches. 8-elt tuple 
    x_start_batches = torch.split(x_start, microbatch_size, dim=0)

    logger.info("Microbatching...")
    logger.info(f"x_start shape: {x_start_batches[0].shape}")
    
    # Split y_n into microbatches (if y_n is a tensor) or handle a list
    if isinstance(y_n, torch.Tensor):
        y_n_batches = torch.split(y_n, microbatch_size, dim=0)
    elif isinstance(y_n, list):
        y_n_batches = [torch.split(y, microbatch_size, dim=0) for y in y_n]
    else:
        raise ValueError("y_n should be a tensor or list of tensors.")

    # Initialize a list to collect the results from each microbatch
    sample_microbatches = []

    # Process each microbatch
    # Assert subframes divisible by microbatch size
    assert subframes % microbatch_size == 0, f"Subframes must be divisible by microbatch size {subframes} % {microbatch_size} != 0"

    div = subframes // microbatch_size
    for i, x_mb in enumerate(x_start_batches):
        # Get the corresponding y_n microbatch
        if isinstance(y_n_batches, tuple):
            # Previously, y_n_batches was a tuple of tensors
            y_mb = y_n_batches[i]
        else:
            y_mb = [y[i] for y in y_n_batches]
        
        measurement_cond_fn = partial(measurement_cond_fn, microbatch_idx=i % div, logger=logger)
        sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)
        
        # Call the sample function on the microbatch
        microbatch_sample = sample_fn(x_start=x_mb.to(device), measurement=y_mb, record=False, save_root=out_path)
        # microbatch_sample = torch.randn(x_mb.shape, device=device)
        
        # Append the result to the list
        sample_microbatches.append(microbatch_sample)

    # Reassemble the final result from the processed microbatches
    sample = torch.cat(sample_microbatches, dim=0)
    
    return sample

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--microbatch_size', type=int, default=1)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    parser.add_argument('--eval', action='store_true')
    args = parser.parse_args()
   
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)

    measure_config = task_config['measurement']
    multi = measure_config.get('multi', False)

    # Working directory
    out_path = os.path.join(args.save_dir,
                             measure_config['operator']['name'] if not multi else "multi"
                            )
    os.makedirs(out_path, exist_ok=True)

    # logger
    # Logfile: out_path/logs/{datetime}.log
    logger = get_logger(os.path.join(out_path, 'logs'))

    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str) 

    #assert model_config['learn_sigma'] == diffusion_config['learn_sigma'], \
    #"learn_sigma must be the same for model and diffusion configuartion."
    
    # Load model
    model = create_model(**model_config)
    model = model.to(device)
    model.eval()

    if args.eval:
        psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)
        psnr_list, ssim_list, lpips_list = [], [], []

    # Prepare Operator and noise
    logger.info(f"Multi: {multi}")  
    if not multi:
        operator = get_operator(device=device, **measure_config['operator'])
        noiser = get_noise(**measure_config['noise'])
        logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")
    else:
        # Multiple operators, each keyed as operator_0, operator_1, ...
        noiser = get_noise(**measure_config['noise'])
        i = 0
        operators = []
        # while operator_i exists, add it to the list of operators
        while f"operator_{i}" in measure_config:
            operators.append(get_operator(device=device, **measure_config[f"operator_{i}"]))
            logger.info(f"Operation: {measure_config[f'operator_{i}']['name']}")
            i += 1
        operator = operators
        

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    img_dirs = ['recon', 'progress', 'label']
    if not multi:
        img_dirs.append('input')
    else:
        for i in range(len(operator)):
            img_dirs.append(f'input_{i}')

    for img_dir in img_dirs:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                    transforms.Resize((256, 256)) # added in case
                                    ])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask 
    if not multi and measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )

    # Check if subframes is defined
    subframes = None
    if multi:
        subframes = measure_config.get('subframes', 1)
    elif measure_config['operator']['name'] == 'coded_inverse':
        subframes = measure_config['operator']['subframes']

    sum_time = 0
    time_count = 0
    # Do Inference
    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        logger.info(f"Shape of the image: {ref_img.shape}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)
        if len(ref_img.shape) == 5:
            logger.info(f"Number of frames: {ref_img.shape[2]}")
            ref_img = rearrange(ref_img, 'b c t h w -> (b t) c h w')
            logger.info(f"Shape of the new image: {ref_img.shape}")

        # If subframes defined, take the first subframes
        logger.info(f"Subframes: {subframes}")
        if subframes is not None:
            ref_img = ref_img[:subframes]

        # Time
        time_count += ref_img.shape[0]
        start = time.time()

        # Exception) In case of inpainging,
        if not multi and measure_config['operator'] ['name'] == 'inpainting':
            mask = mask_gen(ref_img)
            mask = mask[:, 0, :, :].unsqueeze(dim=0)
            measurement_cond_fn = partial(cond_method.conditioning, mask=mask)
            sample_fn = partial(sample_fn, measurement_cond_fn=measurement_cond_fn)

            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img, mask=mask)
            y_n = noiser(y)

        elif multi:
            y_n = []
            coded_image = None
            for j in range(len(operator)):
                if measure_config[f'operator_{j}']['name'] == 'super_resolution': 
                    K = measure_config[f'operator_{j}']['scale_factor']
                    if coded_image is not None:
                        coded_image = coded_image.cpu().detach().numpy()
                        y = reshuffle_mosaic2vid(coded_image, K)
                        # logger.info(f"Super resolution shape: {y.shape}")
                        y = torch.tensor(y, device=device)
                        y = rearrange(y, 'b c t h w -> (b t) c h w')
                    else:
                        raise ValueError("Coded image is not available.")
                elif measure_config[f'operator_{j}']['name'] == 'coded_inverse':
                    assert subframes is not None, "Subframes must be defined for coded inverse."
                    
                    y = operator[j].forward(ref_img)
                    coded_image = torch.sum(y.reshape(-1, subframes, *y.shape[1:]), axis=1)
                else:
                    y = operator[j].forward(ref_img)
                y_n.append(noiser(y))
                logger.info(f"Measurement {j} shape: {y_n[j].shape}")

        else: 
            # Forward measurement model (Ax + n)
            y = operator.forward(ref_img)
            y_n = noiser(y)
            logger.info(f"Measurement shape: {y_n.shape}")

         
        # Sampling
        visualize(os.path.join(out_path, 'label', fname), ref_img)

        if multi:
            for j in range(len(y_n)):
                visualize(os.path.join(out_path, f'input_{j}', fname), y_n[j])
        else:
            visualize(os.path.join(out_path, 'input', fname), y_n)

        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        if 0 < args.microbatch_size < x_start.shape[0]:
            sample = microbatch_sample(x_start, y_n, sample_fn, measurement_cond_fn, args.microbatch_size, device, logger, measure_config.get('subframes', 1), out_path)
        else:
            sample = sample_fn(x_start=x_start, measurement=y_n, record=False, save_root=out_path)
        # sample = torch.randn(ref_img.shape, device=device)
        visualize(os.path.join(out_path, 'recon', fname), sample)

        # Test range and type of ref_image, sample

        if args.eval:
            # [-1, 1] -> [0, 1]
            # then clamp
            eval_sample = torch.clamp((sample + 1) / 2, 0, 1)
            eval_ref_img = torch.clamp((ref_img + 1) / 2, 0, 1)

            psnr_val = psnr(eval_sample, eval_ref_img).item()
            ssim_val = ssim(eval_sample, eval_ref_img).item()
            lpips_val = lpips_fn(eval_sample, eval_ref_img).item()

            psnr_list.append(psnr_val)
            ssim_list.append(ssim_val)
            lpips_list.append(lpips_val)
            
            logger.info(f"PSNR: {psnr_val}")
            logger.info(f"SSIM: {ssim_val}")
            logger.info(f"LPIPS: {lpips_val}")

        end = time.time()
        sum_time += end - start

        logger.info("-"*70)
        logger.info('\n')
    
    logger.info("Inference done.")
    # Print the mean results
    if args.eval:
        logger.info(f"Mean time (per image): {sum_time / time_count}")
        logger.info(f"Mean PSNR: {np.mean(psnr_list)}")
        logger.info(f"Mean SSIM: {np.mean(ssim_list)}")
        logger.info(f"Mean LPIPS: {np.mean(lpips_list)}")
        

if __name__ == '__main__':
    main()
