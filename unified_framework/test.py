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
import glob
import argparse
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity
import h5py
from unet import UNet
import utils
from inverse import ShiftVarConv2D, StandardConv2D
import scipy.io as scio
from dataloader import SixGraySimData, GraySimDavis
from lpips import LPIPS

## parse arguments
# Dirs: /scratch/ondemand28/dsaragih/datasets/TestDAVIS/TestImages, /u8/d/dsaragih/diffusion-posterior-sampling/STFormer/test_datasets/simulation
parser = argparse.ArgumentParser()
parser.add_argument('--savedir', type=str, default='results' ,help='export dir name to dump results')
parser.add_argument('--data_path', type=str, default='/u8/d/dsaragih/diffusion-posterior-sampling/STFormer/test_datasets/simulation', help='path to test data')
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

test_data = SixGraySimData(args.data_path,mask=code_repeat.squeeze(0).cpu().numpy())
data_loader = torch.utils.data.DataLoader(test_data, batch_size=1, shuffle=False)

logging.info('Starting inference')

full_gt = []
full_pred = []
psnr_list, ssim_list, lpips_list = [], [], []
psnr_dict, ssim_dict, lpips_dict = {}, {}, {}
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

        if data_iter > 3:
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
        # Only first.
        for seq in range(batch_size):
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

            logging.info(f"b1 shape: {b1.shape}, interm_vid shape: {interm_vid.shape}, vid shape: {vid.shape}")
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
            # if save_images:
            #     os.makedirs(os.path.join(save_path, _name), exist_ok=True)
            #     os.makedirs(os.path.join(save_path, _name, 'frames'), exist_ok=True)
            #     utils.save_image(b1_np, os.path.join(save_path, _name, 'seq_%.2d_coded.png'%(seq+1)))
            #     if args.two_bucket:
            #         utils.save_image(b0_np, os.path.join(save_path, _name, 'seq_%.2d_complement.png'%(seq+1)))
            #     if args.save_gif:
            #         utils.save_gif(vid_np, os.path.join(save_path, _name, 'seq_%.2d_gt.gif'%(seq+1)))
            #         utils.save_gif(highres_np, os.path.join(save_path, _name, 'seq_%.2d_recon.gif'%(seq+1)))
            #     for sub_frame in range(vid_np.shape[0]):
            #         utils.save_image(highres_np[sub_frame], os.path.join(save_path, _name, 'frames', 'seq_%.2d_recon_%.2d.png'%(seq+1, sub_frame+1)))

        avg_psnr = psnr.compute().item()
        avg_ssim = ssim.compute().item()
        avg_lpips = lpips_fn.compute().item()

        psnr.reset()
        ssim.reset()
        lpips_fn.reset()

        logging.info('Sequence %s done'%(_name))
        logging.info('Average PSNR: %.2f'%(avg_psnr))
        logging.info('Average SSIM: %.3f'%(avg_ssim))
        logging.info('Average LPIPS: %.3f'%(avg_lpips))
        logging.info('Saved images and gifs for all sequences')

        psnr_list.append(avg_psnr)
        ssim_list.append(avg_ssim)
        lpips_list.append(avg_lpips)

        psnr_dict[_name] = avg_psnr
        ssim_dict[_name] = avg_ssim
        lpips_dict[_name] = avg_lpips

    # with h5py.File('predict.h5','w') as write:
    #     full_gt = np.stack(full_gt,0)
    #     full_pred = np.stack(full_pred,0)
    #     print(f'shape of full_gt:{full_gt.shape} and full_pred:{full_pred.shape}')
    #     write.create_dataset('gt',data=full_gt,compression='gzip')
    #     write.create_dataset('pred',data=full_pred,compression='gzip')
    dash_line = '-' * 80 + '\n'
    psnr_dict["psnr_mean"] = np.mean(psnr_list)
    ssim_dict["ssim_mean"] = np.mean(ssim_list)
    lpips_dict["lpips_mean"] = np.mean(lpips_list)
    
    psnr_str = ", ".join([key+": "+"{:.4f}".format(psnr_dict[key]) for key in psnr_dict.keys()])
    ssim_str = ", ".join([key+": "+"{:.4f}".format(ssim_dict[key]) for key in ssim_dict.keys()])
    lpips_str = ", ".join([key+": "+"{:.4f}".format(lpips_dict[key]) for key in lpips_dict.keys()])
    logging.info("Mean PSNR: \n"+
                dash_line + 
                "{}.\n".format(psnr_str)+
                dash_line)

    logging.info("Mean SSIM: \n"+
                dash_line + 
                "{}.\n".format(ssim_str)+
                dash_line) 
    logging.info("Mean LPIPS: \n"+
                dash_line + 
                "{}.\n".format(lpips_str)+
                dash_line)

logging.info('Finished inference')