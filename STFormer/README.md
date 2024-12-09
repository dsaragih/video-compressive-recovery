# STFormer for video SCI
This repo is the implementation of "[Spatial-Temporal Transformer for Video Snapshot Compressive Imaging](https://arxiv.org/abs/2209.01578)". 
## Abstract
 Video snapshot compressive imaging (SCI) captures multiple sequential video frames by a single measurement using the idea of computational imaging. The underlying principle is to modulate high-speed frames through different masks and these
modulated frames are summed to a single measurement captured by a low-speed 2D sensor (dubbed optical encoder); following this, algorithms are employed to reconstruct the desired high-speed frames (dubbed software decoder) if needed. In this paper, we consider the reconstruction algorithm in video SCI, i.e., recovering a series of video frames from a compressed measurement. Specifically, we propose a Spatial-Temporal transFormer (STFormer) to exploit the correlation in both spatial and temporal domains. STFormer network is composed of a token generation block, a video reconstruction block, and these two blocks are connected by a series of STFormer blocks. Each STFormer block consists of a spatial self-attention branch, a temporal self-attention branch and the outputs of these two branches are integrated by a fusion network. Extensive results on both simulated and real data demonstrate the state-of-the-art performance of STFormer. 
## Testing Result on Simulation Dataset
<div align="center">
  <img src="docs/gif/Bosphorus.gif" />  
  <img src="docs/gif/ShakeNDry.gif" />  

  Fig1. Reconstructed Color Data via Different Algorithms
</div>

## Installation
Please see the [Installation Manual](docs/install.md) for STFormer Installation. 


## Training 
Support multi GPUs and single GPU training efficiently. First download DAVIS2017 dataset from the [DAVIS website](https://davischallenge.org/davis2017/code.html); we use the entirety of TrainVal for training and Test-Dev2017 for testing. Next, modify the *data_root* value in the relevant configuration files: these should be `configs/STFormer/davis.py` and `configs/STFormer/davis_test.py`.

Launch multi GPU training by the statement below:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4  --master_port=3278 tools/train.py configs/STFormer/stformer_base.py --distributed=True
```

Launch single GPU training by the statement below.

Default using GPU 0. One can also choosing GPUs by specify CUDA_VISIBLE_DEVICES

```
python tools/train.py configs/STFormer/stformer_base.py
```

## Testing STFormer on Grayscale Simulation Dataset 
Specify the path of weight parameters, then launch 6 benchmark test in grayscale simulation dataset by executing the statement below.

```
python tools/test.py configs/STFormer/stformer_base.py --weights=checkpoints/davis_2.pth
```

## Testing STFormer on DAVIS2017 Test Dataset 
Specify the path of weight parameters, then launch the DAVIS2017 test by executing the statement below.

```
python tools/test_davis.py configs/STFormer/stformer_base_4.py --weights=checkpoints/davis_4.pth --work_dir "work_dirs/mask_4_epoch_45"
```

## Citation
```
@article{wang2023spatial,
  author={Wang, Lishun and Cao, Miao and Zhong, Yong and Yuan, Xin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Spatial-Temporal Transformer for Video Snapshot Compressive Imaging}, 
  year={2023},
  volume={45},
  number={7},
  pages={9072-9089},
  doi={10.1109/TPAMI.2022.3225382}}
```
## Acknowledgement
The codes are based on [CACTI](https://github.com/ucaswangls/cacti), 
we also refer to codes in [Swin Transformer](https://github.com/microsoft/Swin-Transformer.git), 
[Video Swin Transformer](https://github.com/SwinTransformer/Video-Swin-Transformer), 
[RevSCI](https://github.com/BoChenGroup/RevSCI-net) 
and [Two Stage](https://arxiv.org/pdf/2201.05810). Thanks for their awesome works.
