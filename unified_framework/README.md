# A Unified Framework for Compressive Video Recovery from Coded Exposure Techniques

This repository contains the official implementation of the work **A Unified Framework for Compressive Video Recovery from Coded Exposure Techniques** accepted to be published at IEEE/CVF WACV 2021.

## args used in evaluating the model

- ```--ckpt``` : refers to the name of the model to be used
- ```--save_gif``` : saves the ground truth and predicted frames to the disk. Otherwise the code only logs the PSNR values.
- ```--flutter``` : to be used when evaluating the flutter shutter model
- ```--two_bucket``` : to be used when evaluating the two bucked coded-blurred image pair model
- ```--mask_path``` : path to the mask file to be used for evaluation. Default is the mask used in the paper.
- ```--blocksize```: tile block size to be used for evaluation. From the report, we used k = 2, 4, 6, 8.
- ```--subframes```: number of subframes to be used for evaluation. This should be blocksize^2.

## Evaluation
We provide evaluation code for three different and important models in the paper.
We provide the DNN test model in the `models` directory, and the main test scripts are `test.py` and `test_davis.py`. The main difference lies in the data loader used and the `data_path` set. Note that for `test.py` we're using the same simulation dataset as STFormer.

Execute the test script with the following command:
```
python test.py \
    --ckpt "model_{k}.pth" \
    --gpu 0 \
    --two_bucket \
    --blocksize {k} \
    --subframes {k*k} \
    --mask_path "./data/{k}x{k}_mask.mat" \
    --savedir "results"
```
