import cv2
import torch
import lpips
import numpy as np
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
import os

def get_metrics(gt_img, pred_img, lpips_loss_fn):
    

    if gt_img.shape != pred_img.shape:
        pred_img = cv2.resize(pred_img, (gt_img.shape[1], gt_img.shape[0]))
        print(f"Resized prediction image to shape {pred_img.shape}")

    psnr_score = psnr(gt_img, pred_img, data_range=255)
    ssim_score = ssim(gt_img, pred_img, data_range=255, multichannel=False)

    gt_img = np.stack([gt_img] * 3, axis=-1)
    pred_img = np.stack([pred_img] * 3, axis=-1)
                            
    # Normalize images from [0, 255] to [-1, 1]
    # print(f"Max, min of gt: {gt_img.max()}, {gt_img.min()}")
    # print(f"Max, min of pred: {pred_img.max()}, {pred_img.min()}")
    gt_img = (gt_img / 255.0) * 2 - 1
    pred_img = (pred_img / 255.0) * 2 - 1

    # Convert from numpy arrays to PyTorch tensors
    gt_img = torch.from_numpy(gt_img).permute(2, 0, 1).unsqueeze(0)  # shape: (1, 3, H, W)
    pred_img = torch.from_numpy(pred_img).permute(2, 0, 1).unsqueeze(0)  # shape: (1, 3, H, W)

    lpips_score = lpips_loss_fn(gt_img, pred_img).squeeze().item()

    return {'psnr': psnr_score,
            'ssim': ssim_score, 
            'lpips': lpips_score}


if __name__ == "__main__":
    loss_fn = lpips.LPIPS(net='vgg')
    k = 4
    gt_dir = f"cacti/00001"
    list_eval = [
        f"eval_imgs/crash_{k}x{k}"
    ]

    # Dictionary to store metrics for each evaluation directory
    metrics_dict = {os.path.basename(eval_dir): [] for eval_dir in list_eval}

    # Loop through all images in the directories (in sorted order)
    for i, gt_img_name in enumerate(sorted(os.listdir(gt_dir))):
        gt_img = cv2.imread(os.path.join(gt_dir, gt_img_name), 0).astype(np.float32)
        
        # Loop over each evaluation directory
        for eval_dir in list_eval:
            eval_dir_name = os.path.basename(eval_dir)
            eval_img_name = sorted(os.listdir(eval_dir))[i]  # Corresponding image
            eval_img = cv2.imread(os.path.join(eval_dir, eval_img_name), 0).astype(np.float32)
            
            # Calculate and store the metric for this image pair
            metrics_dict[eval_dir_name].append(get_metrics(gt_img, eval_img, loss_fn))
            
        print(f"Finished processing image {i}")

    for eval_dir_name, metrics in metrics_dict.items():
        print(f"\n{eval_dir_name.split('/')[-1]} METRICS")
        print(metrics)
        # Calculate and print the average of each metric
        avg_psnr = np.mean([m['psnr'] for m in metrics])
        avg_ssim = np.mean([m['ssim'] for m in metrics])
        avg_lpips = np.mean([m['lpips'] for m in metrics])
        
        print(f"Average PSNR: {avg_psnr}")
        print(f"Average SSIM: {avg_ssim}")
        print(f"Average LPIPS: {avg_lpips}")


