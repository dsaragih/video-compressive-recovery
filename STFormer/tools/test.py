import os
import os.path as osp
import sys 
BASE_DIR = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.append(BASE_DIR)
import torch 
from torch.utils.data import DataLoader
from cacti.utils.mask import generate_masks
from cacti.utils.utils import save_single_image,get_device_info,load_checkpoints, interpolate_mosaic2vid, print_mean_metrics
from cacti.utils.config import Config
from cacti.models.builder import build_model
from cacti.datasets.builder import build_dataset 
from cacti.utils.logger import Logger
from torch.cuda.amp import autocast
import numpy as np 
import argparse 
import time
import einops 

from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config",type=str)
    parser.add_argument("--work_dir",type=str)
    parser.add_argument("--weights",type=str)
    parser.add_argument("--device",type=str,default="cuda:0")
    args = parser.parse_args()
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    if not torch.cuda.is_available():
        args.device="cpu"
    return args
    

def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)

    device = args.device
    config_name = osp.splitext(osp.basename(args.config))[0]
    if args.work_dir is None:
        model_name = args.weights.split("/")[-1].split(".")[0]
        args.work_dir = osp.join('./work_dirs', model_name)
    mask,mask_s = generate_masks(cfg.test_data.mask_path,cfg.test_data.mask_shape)
    cr = mask.shape[0]
    if args.weights is None:
        args.weights = cfg.checkpoints

    test_dir = osp.join(args.work_dir,"test_images")

    log_dir = osp.join(args.work_dir,"test_log")
    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    logger = Logger(log_dir)

    dash_line = '-' * 80 + '\n'
    device_info = get_device_info()
    env_info = '\n'.join(['{}: {}'.format(k,v) for k, v in device_info.items()])
    logger.info('GPU info:\n' 
            + dash_line + 
            env_info + '\n' +
            dash_line) 
    logger.info(f"Config:\n{cfg.test_data}\n")
    test_data = build_dataset(cfg.test_data,{"mask":mask})
    data_loader = DataLoader(test_data,batch_size=1,shuffle=False,num_workers=4)

    model = build_model(cfg.model).to(device)
    logger.info("Load pre_train model...")
    logger.info(f"Load model from {args.weights}")
    resume_dict = torch.load(args.weights)
    if "model_state_dict" not in resume_dict.keys():
        model_state_dict = resume_dict
    else:
        model_state_dict = resume_dict["model_state_dict"]
    load_checkpoints(model,model_state_dict,strict=True)

    Phi = einops.repeat(mask,'cr h w->b cr h w',b=1)
    Phi_s = einops.repeat(mask_s,'h w->b 1 h w',b=1)
    Phi = torch.from_numpy(Phi).to(args.device)
    Phi_s = torch.from_numpy(Phi_s).to(args.device)

    logger.info(f"Phi shape: {Phi.shape}")
    logger.info(f"Phi_s shape: {Phi_s.shape}")

    if "partition" in cfg.test_data.keys():
        logger.info("Chunking image...")
        partition = cfg.test_data.partition
        _,_,Phi_h,Phi_w = Phi.shape
        part_h = partition.height
        part_w = partition.width
        assert (Phi_h%part_h==0) and (Phi_w%part_w==0), "Image cannot be chunked!"
        h_num = Phi_h//part_h
        w_num = Phi_w//part_w
        A_Phi = einops.rearrange(Phi,"b cr (h_num h) (w_num w)->(b h_num w_num) cr h w",h=part_h,w=part_w)
        A_Phi_s = einops.rearrange(Phi_s,"b cr (h_num h) (w_num w)->(b h_num w_num) cr h w",h=part_h,w=part_w)
        
    psnr_dict,ssim_dict,lpips_dict = {},{},{}
    psnr_li_dict,ssim_li_dict,lpips_li_dict = {},{},{}

    # Metrics
    psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)

    psnr_li = PeakSignalNoiseRatio(data_range=1.0).to(device)
    ssim_li = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
    lpips_li = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(device)

    sum_time=0.0
    time_count = 0
    save_images = True
    for data_iter,data in enumerate(data_loader):
        batch_output = []
        li_output = []
        meas, gt = data

        # if gt all zeros, skip
        if torch.sum(gt) == 0:
            continue

        if data_iter > 30:
            # break
            save_images = False

        # Index batch dimension
        gt = gt[0].numpy()
        meas = meas[0].float().to(device)
        batch_size = meas.shape[0] # frames / subframes
        name = test_data.data_name_list[data_iter]
        if "_" in name:
            _name,_ = name.split("_")
        else:
            _name,_ = name.split(".")
        out_list = []
        gt_list = []

        for j in range(gt.shape[0]):
            gt_dir = osp.join(test_dir,_name+"_gt")
            if not osp.exists(gt_dir):
                os.makedirs(gt_dir)

            # Min, max
            logger.info(f"GT min: {np.min(gt[j])}, GT max: {np.max(gt[j])}")
            save_single_image(gt[j],gt_dir,j,name=config_name)  

        for ii in range(batch_size):
            single_gt = gt[ii]
            single_meas = meas[ii].unsqueeze(0).unsqueeze(0)
            if "partition" in cfg.test_data.keys():
                Phi = A_Phi[ii%(h_num*w_num)].unsqueeze(0)
                Phi_s = A_Phi_s[ii%(h_num*w_num)].unsqueeze(0)
            with torch.no_grad():
                torch.cuda.synchronize()
                start = time.time()
                if "amp" in cfg.keys() and cfg.amp:
                    with autocast():
                        outputs = model(single_meas, Phi, Phi_s)
                else:
                    outputs = model(single_meas, Phi, Phi_s)
                torch.cuda.synchronize()
                end = time.time()
                run_time = end - start
                if ii>0:
                    sum_time += run_time
                    time_count += 1
            if not isinstance(outputs,list):
                outputs = [outputs]
            output = outputs[-1][0].cpu().numpy().astype(np.float32)
            
            if "partition" in cfg.test_data.keys():
                out_list.append(output)
                gt_list.append(single_gt)
                if (ii+1)%(h_num*w_num)==0:
                    output = np.array(out_list)
                    single_gt = np.array(gt_list)
                    output = einops.rearrange(output,"(h_num w_num) c cr h w->c cr (h_num h) (w_num w)",h_num=h_num,w_num=w_num)
                    single_gt = einops.rearrange(single_gt,"(h_num w_num) cr h w->cr (h_num h) (w_num w)",h_num=h_num,w_num=w_num)
                    batch_output.append(output)
                    out_list = []
                    gt_list = []
                else:
                    continue
            else:
                batch_output.append(output)
                # Linear interpolation
                lin = interpolate_mosaic2vid(meas[ii].detach().cpu().numpy(), mask)
                li_output.append(lin)

        first_gt = torch.tensor(gt[0:1]).to(device)
        first_lin = torch.tensor(li_output).to(device)
        output_tensor = torch.tensor(batch_output).to(device)

        # logger.info(f"Shape of output tensor: {output_tensor.shape}")
        # logger.info(f"Shape of first_gt: {first_gt.shape}")
        # logger.info(f"Shape of first_lin: {first_lin.shape}")

        # (b x t x c x h x w) -> (bt x c x h x w)
        output_tensor = output_tensor.view(-1, *output_tensor.shape[2:])
        first_gt = first_gt.view(-1, *first_gt.shape[2:])
        first_lin = first_lin.view(-1, *first_lin.shape[2:])

        # Add channel dim if missing (t x h x w -> t x 3 x h x w)
        # use cv2 to convert to 3 channel image
        if output_tensor.ndim == 3:
            output_tensor = torch.stack([output_tensor]*3, dim=1)
            first_gt = torch.stack([first_gt]*3, dim=1)
            first_lin = torch.stack([first_lin]*3, dim=1)

        # Clamp to [0, 1]
        output_tensor = torch.clamp(output_tensor, 0.0, 1.0).float()
        first_gt = torch.clamp(first_gt, 0.0, 1.0).float()
        first_lin = torch.clamp(first_lin, 0.0, 1.0).float()
        
        # Check shapes
        assert first_gt.shape == output_tensor.shape == first_lin.shape, f"Shapes do not match: {first_gt.shape} vs {output_tensor.shape} vs {first_lin.shape}"
        # Metrics
        psnr_dict[_name] = psnr(output_tensor, first_gt)
        ssim_dict[_name] = ssim(output_tensor, first_gt)
        lpips_dict[_name] = lpips_fn(output_tensor, first_gt)

        psnr_li_dict[_name] = psnr_li(first_lin, first_gt)
        ssim_li_dict[_name] = ssim_li(first_lin, first_gt)
        lpips_li_dict[_name] = lpips_li(first_lin, first_gt)

        logger.info("{}, Mean PSNR: {:.4f} Mean SSIM: {:.4f} Mean LPIPS: {:.4f}.".format(_name,psnr_dict[_name],ssim_dict[_name],lpips_dict[_name]))
        logger.info("{}, Linear Interpolation, Mean PSNR: {:.4f} Mean SSIM: {:.4f} Mean LPIPS: {:.4f}.".format(_name,psnr_li_dict[_name],ssim_li_dict[_name],lpips_li_dict[_name]))

        #save image
        if save_images:
            out = np.array(batch_output)
            li_out = np.array(li_output)
            for j in range(out.shape[0]):
                image_dir = osp.join(test_dir,_name)
                li_image_dir = osp.join(test_dir,_name+"_li")
                if not osp.exists(image_dir):
                    os.makedirs(image_dir)
                if not osp.exists(li_image_dir):
                    os.makedirs(li_image_dir)
                save_single_image(out[j],image_dir,j,name=config_name)
                save_single_image(li_out[j],li_image_dir,j,name=config_name)

    if time_count==0:
        time_count=1
    logger.info('Average Run Time:\n' 
            + dash_line + 
            "{:.4f} s.".format(sum_time/time_count) + '\n' +
            dash_line)
    
    print_mean_metrics('PSNR', psnr_dict,logger)
    print_mean_metrics('SSIM', ssim_dict,logger)
    print_mean_metrics('LPIPS', lpips_dict,logger)

    print_mean_metrics('PSNR LI', psnr_li_dict,logger)
    print_mean_metrics('SSIM LI', ssim_li_dict,logger)
    print_mean_metrics('LPIPS LI', lpips_li_dict,logger)

if __name__=="__main__":
    main()
