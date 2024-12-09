'''
------------------------------------------------------
RUN INFERENCE ON SAMPLE TEST SEQUENCES - ATTENTION NET
------------------------------------------------------
'''
import os 
import numpy as np
import torch
import torch.nn.functional as F
import logging
from glob import glob
import argparse
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

import torchvision
from PIL import Image
import cv2

from unet import UNet
import utils
from inverse import ShiftVarConv2D, StandardConv2D
import scipy.io as scio
from dataloader import SixGraySimData, GraySimDavis
import einops 


## parse arguments
# Dirs: /scratch/ondemand28/dsaragih/datasets/TestDAVIS/TestImages, /u8/d/dsaragih/diffusion-posterior-sampling/STFormer/test_datasets/simulation
parser = argparse.ArgumentParser()
parser.add_argument('--savedir', type=str, default='results' ,help='export dir name to dump results')
parser.add_argument('--data_path', type=str, default='/scratch/ondemand28/dsaragih/datasets/TestDAVIS/TestImages', help='path to test data')
parser.add_argument('--ckpt', type=str, required=True, help='checkpoint full name')
parser.add_argument('--blocksize', type=int, default=2, help='tile size for code default 3x3')
parser.add_argument('--subframes', type=int, default=4, help='num sub frames')
parser.add_argument('--gpu', type=str, required=True, help='GPU ID')
parser.add_argument('--mask_path', type=str, default='data/mask_2x2.mat', help='mask path')
parser.add_argument('--two_bucket', action='store_true', help='1 bucket or 2 buckets')
parser.add_argument('--save_gif', action='store_true', help='saves gifs')
parser.add_argument('--log_interval', type=int, default=1, help='save gifs interval')
parser.add_argument('--intermediate', action='store_true', help="intermediate reconstruction")
parser.add_argument('--flutter', action='store_true', help="for flutter shutter")

args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

save_path = os.path.join(args.savedir,args.ckpt.split('.')[0])
if not os.path.exists(save_path):
  os.mkdir(save_path)
  os.mkdir(os.path.join(save_path, 'frames'))


## configure runtime logging
logging.basicConfig(level=logging.INFO,
                    filename=os.path.join(save_path, 'logfile.log'), 
                    format='%(asctime)s - %(message)s', 
                    filemode='w')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
console.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logging.getLogger().addHandler(console)
logging.info(args)


## load net
if args.intermediate:
    num_features = args.subframes
else:
    num_features = 64
if args.flutter:
    assert not args.two_bucket
    invNet = StandardConv2D(out_channels=num_features).cuda()
else:
    invNet = ShiftVarConv2D(out_channels=num_features, block_size=args.blocksize, two_bucket=args.two_bucket).cuda()
uNet = UNet(in_channel=num_features, out_channel=args.subframes, instance_norm=False).cuda()


## load checkpoint
ckpt = torch.load(os.path.join('models', args.ckpt))
invNet.load_state_dict(ckpt['invnet_state_dict'])
uNet.load_state_dict(ckpt['unet_state_dict'])
invNet.eval()
uNet.eval()

# Mask
mask = scio.loadmat(args.mask_path)['mask']
mask = np.transpose(mask, [2, 0, 1])
mask = mask.astype(np.float32)
input_params = {'height': mask.shape[-2],
                'width': mask.shape[-1]}
# Crop block size
# c2b_code = mask[:, :args.blocksize, :args.blocksize].unsqueeze(0) # 1 x 4 x 2 x 2
c2b_code = ckpt['c2b_state_dict']['code']
code_repeat = c2b_code.repeat(1, 1, input_params['height']//args.blocksize, input_params['width']//args.blocksize) # 1 x 4 x 256 x 256

print(f"Input height: {input_params['height']}, width: {input_params['width']}")
print(f"c2b_code shape: {c2b_code.shape}")
print(f"Code repeat shape: {code_repeat.shape}")

# Save code repeat
# for i in range(code_repeat.shape[1]):
#     utils.save_image(code_repeat[0, i].cpu().numpy(), os.path.join(save_path, f'code_{i}.png'))

test_data = GraySimDavis(args.data_path,mask=code_repeat.squeeze(0).cpu().numpy())
data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

logging.info('Starting inference')

full_gt = []
full_pred = []
psnr = PeakSignalNoiseRatio(data_range=1.0).cuda()
ssim = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).cuda()

save_images = True
with torch.no_grad():
    for data_iter,data in enumerate(data_loader):
        meas, gt = data
        # if gt all zeros, skip
        if torch.sum(gt) == 0:
            continue

        if data_iter > 300:
            save_images = False

        gt = gt[0].numpy() # (f/s, s, H,W)
        meas = meas[0].float().cuda() # (f/s, 1 ,H,W)
        batch_size = meas.shape[0] # frames / subframes
        name = test_data.data_name_list[data_iter]
        if "_" in name:
            _name,_ = name.split("_")
        else:
            _name,_ = name.split(".")
            
        psnr_sum = 0.
        ssim_sum = 0.
        lpips_sum = 0.

        # Only first sequence
        for seq in range(1):
            vid = torch.cuda.FloatTensor(gt[seq:seq+1,...]) # (1, s, H,W)
            # Number of frames might be less than code_repeat
            # If vid not [image_height, image_width], pad to [image_height, image_width]
            pad = (0, input_params['height'] - vid.shape[-2], 0, input_params['width'] - vid.shape[-1])
            vid = F.pad(vid, pad, 'constant', 0)
            b1 = meas[seq:seq+1,...] # (1, 1, H,W)
            b1 = F.pad(b1, pad, 'constant', 0)
            if not args.two_bucket:
                interm_vid = invNet(b1) 
            else:
                b0 = torch.mean(vid, dim=1, keepdim=True)
                b_stack = torch.cat([b1,b0], dim=1)
                interm_vid = invNet(b_stack)
                if interm_vid.shape[-1] != vid.shape[-1]:
                    interm_vid = F.pad(interm_vid, (0, vid.shape[-1]-interm_vid.shape[-1], 0, vid.shape[-2]-interm_vid.shape[-2]))

            # logging.info(f"b1 shape: {b1.shape}, interm_vid shape: {interm_vid.shape}, vid shape: {vid.shape}")
            highres_vid = uNet(interm_vid) # (1,16,H,W)
            
            assert highres_vid.shape == vid.shape, f"Highres vid shape: {highres_vid.shape}, vid shape: {vid.shape}"
            highres_vid = torch.clamp(highres_vid, min=0, max=1)
            
            ## converting tensors to numpy arrays
            b1_np = b1.squeeze().data.cpu().numpy() # (H,W)
            if args.two_bucket:
                b0_np = b0.squeeze().data.cpu().numpy()
            vid_np = vid.squeeze().data.cpu().numpy() # (16,H,W)
            highres_np = highres_vid.squeeze().data.cpu().numpy() # (16,H,W)
            full_pred.append(highres_np)
            full_gt.append(vid_np)

            if seq == 0 and data_iter == 0:
                logging.info('Shapes: b1_np: %s, vid_np: %s, highres_np: %s'%(b1_np.shape, vid_np.shape, highres_np.shape))
                logging.info('Min: b1_np: %s, vid_np: %s, highres_np: %s'%(b1_np.min(), vid_np.min(), highres_np.min()))
                logging.info('Max: b1_np: %s, vid_np: %s, highres_np: %s'%(b1_np.max(), vid_np.max(), highres_np.max()))

            # Check they are the same size
            assert vid_np.shape == highres_np.shape, f"Shapes do not match: {vid_np.shape} vs {highres_np.shape}"

            first_gt = torch.tensor(vid_np).cuda()
            output_tensor = torch.tensor(highres_np).cuda()

            # Add channel dim if missing (t x h x w -> t x 3 x h x w)
            # use cv2 to convert to 3 channel image
            if output_tensor.ndim == 3:
                output_tensor = torch.stack([output_tensor]*3, dim=1)
                first_gt = torch.stack([first_gt]*3, dim=1)

            # Clamp to [0, 1]
            output_tensor = torch.clamp(output_tensor, 0.0, 1.0).float()
            first_gt = torch.clamp(first_gt, 0.0, 1.0).float()
            
            # Check shapes
            assert first_gt.shape == output_tensor.shape, f"Shapes do not match: {first_gt.shape} vs {output_tensor.shape}"

            # Metrics
            psnr.update(output_tensor, first_gt)
            ssim.update(output_tensor, first_gt)
            lpips_fn.update(output_tensor, first_gt)


            ## saving images and gifs
            if save_images:
                os.makedirs(save_path, exist_ok=True)
                base_count = len(glob(os.path.join(save_path, "*.gif")))
                video_path = os.path.join(save_path, f"video_{base_count:02d}.gif")

                vid = (
                    (einops.rearrange(output_tensor, "t c h w -> t h w c") * 255)
                    .cpu()
                    .numpy()
                    .astype(np.uint8)
                )
                # write video file
                # logger.info(f"Writing video shape {vid.shape} to {video_path}")
                frames = [Image.fromarray(frame) for frame in vid]

                # Save as a GIF
                frames[0].save(
                    video_path,
                    save_all=True,
                    append_images=frames[1:],  # Add subsequent frames
                    duration=200,  # Frame duration in milliseconds (e.g., 5 fps = 200 ms)
                    loop=0  # Infinite loop
                )

                # write ground truth and sampled images
                rows = np.sqrt(vid.shape[0]).astype(int)
                gt_grid = torchvision.utils.make_grid(first_gt, nrow=rows)
                gt_grid = gt_grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                gt_grid = gt_grid.cpu().numpy()
                gt_grid = (gt_grid * 255).astype(np.uint8)
                gt_filename = "gt_{:02d}.jpg".format(base_count)
                cv2.imwrite(os.path.join(save_path, gt_filename), gt_grid)

                samples_grid = torchvision.utils.make_grid(output_tensor, nrow=rows)
                samples_grid = samples_grid.transpose(0, 1).transpose(1, 2).squeeze(-1)
                samples_grid = samples_grid.cpu().numpy()
                samples_grid = (samples_grid * 255).astype(np.uint8)
                samples_filename = "sample_{:02d}.jpg".format(base_count)
                cv2.imwrite(os.path.join(save_path, samples_filename), samples_grid)

    logging.info("PSNR: {:.3f}".format(psnr.compute()))
    logging.info("SSIM: {:.3f}".format(ssim.compute()))
    logging.info("LPIPS: {:.3f}".format(lpips_fn.compute()))



logging.info('Finished inference')