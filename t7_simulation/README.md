# T7 Simulation Pipeline

In this library, we provide a simulation pipeline in order to accurately model the T6, a next-generation computational camera developed by the University of Toronto. The T6 features a dual-tap design, and allows for per-pixel encoded subexposures. The goal of this simulation model is to provide users with an efficient way to accurately prototype the camera’s functionality and results, for use in a variety of applications. 

Here, we overview the steps required to capture training data, learn the camera's noise parameters, and run the simulation.

## Installation

### Quick start
Clone the repo:
```
git clone https://github.com/zachsalehe/t6_simulation
```

Create an environment:
```
conda create -n t6_sim
conda activate t6_sim
pip install -r requirements.txt
```

Quick run commands (details on each command below):
```
python format_data.py --data_root data
python train_gan.py --data_root data
python simulate.py \
   --params final_T7_params.mat \
   --mask masks/t7_coded_exposure_2x2.bmp \
   --input_imgs inputs/4_frames/?.png \
   --mode multi_in_single_out \
   --output_fname outputs/fan_2x2_video 

python reshuffle.py \
    --image_path "./../outputs/fan_4x4_video.png" \
    --output_dir "./../outputs/reshuffled/" 
```

### List of dependencies
```
numpy
scipy
matplotlib
opencv-python
torch
torchvision
lpips
```

### How to run the different files in the directory, and what each of them does
`format_data.py` reformats raw T6 data for training use. It takes one argument, `--data_root`, which is the root to the data directory. Instructions for how to capture experimental data and how to properly store it can be found below.

`train_gan.py` trains a generative adversarial network (GAN) to learn the T6's noise parameters. It also takes one argument, `--data_root`, which is the root to the same data directory as before.

`simulate.py` runs the T6 simulation with its given arguments:
```
--params PARAMS       the path to the parameters .mat file
--mask MASK           the path to the mask .bmp file
--input_imgs INPUT_IMGS
                     the path to the input image(s) (for multiple images,
                     use ? [e.g. images/?.png])
--output_dir OUTPUT_DIR
                     the path to the output directory
--mode MODE           options: single_in (uses a single input image),
                     multi_in_single_out (uses multiple input images,
                     switching at each subframe), multi_in_multi_out (uses
                     multiple input images, switching at each capture)
--num_burst NUM_BURST
                     number of burst captures (only used for single_in and
                     multi_in_single_out modes)
--subframes SUBFRAMES
                     number of subframes per capture (only used for
                     single_input and multi_in_multi_out modes)
--exp EXP             exposure time [us] of each subframe (>= 26.21)
--scale SCALE         linear scale that can be used to scale input images
                     prior to simulation
```
Though every T6 sensor is expected to have varying noise parameters, we provide an example set of parameters stored in `data/params.mat`. We also provide several masks stored in `masks/`.

## Noise modelling
Our noise model takes inspiration from Sam Hasinoff's image formation model, as well as Kristina Monakhova's physics-based noise model (references below).

Sam Hasinoff's image formation model is expressed by $I = \min \set{ \Phi t / g + I_0 + n, I_\max }$, where $\text{Var}(n) = \Phi t / g^2 + \sigma^2_\text{read} / g^2 + \sigma^2_\text{ADC}$. Here, $I$ is the measured pixel value, $\Phi t$ is the number of photons collected over an exposure time $t$, $g$ is the sensor's gain, $I_0$ is the sensor's black level, $n$ is the total additive noise (composed of shot, read, and ADC noise), and $I_\max$ is the sensor's saturation point.

Kristina Monakhova's physics-based noise model is expressed by $N = N_s + N_r + N_{row} + N_{row,t} + N_q + N_f + N_p$. Here, $N$ is the total additive noise, where $N_s$, $N_r$, $N_{row}$, $N_{row,t}$, $N_q$, $N_f$, and $N_p$ respectively represent the added shot noise, read noise, row noise, temporal row noise, quantization noise, fixed pattern noise, and periodic noise.

### Theory
Here, we briefly overview the various learned components of our noise model. 

Consider that our clean input image is $I_{in}$ $[DN]$ (clipped to a 12-bit range), our camera's saturation level is $I_{max}$ $[DN]$, and our simulated exposure time is $t$ $[\mu s]$.

#### Gain
Let $g$ be our constant per-pixel gain parameter. We multiply $I_{in}$ by $g$ in order to account for gain variations across the sensor in our final noisy result.

#### Fixed pattern noise (FPN)
Let $h$ be our constant per-pixel FPN parameter. We add $h$ to our final noisy result.

#### Shot noise
Shot noise normally follows a Poisson distribution with respect to the number of arrived photons. In order to learn our shot noise parameter, we instead choose to approximate this with a Gaussian distribution, which is differentiable with respect to its mean and variance. Let $\lambda_{shot}$ be our sensor-wide shot noise parameter. If our sensor is not saturated (i.e. $I_{in} \cdot g + h < I_{max}$), we add $N_{shot} \sim \mathcal{N}(\mu = 0, \sigma^2 = \lambda_{shot}^2 \cdot I_{in} \cdot g^2)$ to our final noisy result. Otherwise, we simply add $N_{shot} = 0$.

#### Read noise
Let $\lambda_{read}$ be our per-pixel read noise parameter. We add $N_{read} \sim \mathcal{N}(\mu = 0, \sigma^2 = \lambda_{read}^2)$ to our final noisy result.

#### Row noise
Let $\lambda_{row}$ be our sensor-wide row noise parameter. We add $N_{row} \sim \mathcal{N}(\mu = 0, \sigma^2 = \lambda_{row}^2)$ to our final noisy result. Note that each row shares the same Gaussian distributed random variable.

#### Temporal row noise
Let $\lambda_{row_t}$  be our sensor-wide temporal row noise parameter. We add $N_{row_t} \sim \mathcal{N}(\mu = 0, \sigma^2 = \lambda_{row_t}^2)$ to our final noisy result. Note that, in addition to each row sharing the same Gaussian distributed random variable, each image in a consecutive burst also shares the same variables.

#### Quantization noise
Let $\lambda_{quant}$ be our sensor-wide quantization noise parameter. We add $N_{quant} \sim \mathcal{U}(\frac{-\lambda_{quant}}{2}, \frac{\lambda_{quant}}{2})$ to our final noisy result.

#### Dark current
Like shot noise, dark current also follows a Poisson distribution, but instead with respect to time. For the same reasons as before, we also choose to approximate this with a Gaussian distribution. Let $\lambda_{dark}$ be our per-pixel dark current parameter. We add $N_{dark} \sim \mathcal{N}(\mu = \lambda_{dark}^2 \cdot t, \sigma^2 = \lambda_{dark}^2 \cdot t)$ to our final noisy result.

#### Final noisy result
Putting all of our noise sources together, we get out final noisy result, $I_{out}$:
$$I_{out} = I_{in} \cdot g + h + N_{shot} + N_{read} + N_{row} + N_{row_t} + N_{quant} + N_{dark}$$

### Experimental captures
<p align="center">
  <img src=docs/images/experiment.png>
</p>

The figure above demonstrates an example capture setup that can be used to gather experimental data. It is important that our scene is illuminated with natural sunlight, as most conventional lights tend to flicker, which can cause unwanted variations in the captured data. It is also important that we are photographing a plain solid surface, as to eliminate any spatial variations that this may cause in the data. To further reduce spatial variations, the lens cap can also be removed during captures.

The following steps must be done twice; once using an all white mask pattern (isolates left tap), and once using an all black mask pattern (isolates right tap). We will capture 256 photos with 1 subframe each at 50 different exposure times. The exposure times will begin from the T6's minimum exposure setting (26.21 $\mu s$), and increase uniformly until we reach a point where most of our pixels are close to (but not at) their saturation limit. This uniform jump in exposure varies depending on the scene's current illumination. Lastly, we will capture another 256 photos at a much higher exposure time, where each pixel is fully saturated, for the purpose of measuring each pixel's saturation limit. Again, the exposure time required for this will vary depending on the illumination.

```
├── data
│   ├── left_raw
│   │   ├── exp000026.21
│   │   │   ├── 0000.npy
│   │   │   ├── 0001.npy
│   │   │   ├── ....
│   │   │   ├── 0255.npy
│   │   │   ├── black_img.npy
│   │   ├── expXXXXXX.XX
│   │   ├── ...
│   │   ├── expXXXXXX.XX
│   │   ├── saturated
│   │   │   ├── 0000.npy
│   │   │   ├── 0001.npy
│   │   │   ├── ....
│   │   │   ├── 0255.npy
│   │   │   ├── black_img.npy
│   ├── right_raw
│   │   ├── ...
```

The file tree above dictates how experimental data should be stored so it can be properly accessed. Left tap and right tap captures should be saved in folders named `left_raw` and `right_raw` respectively. The content format within each of them remains the same. Captures for any of the 50 exposure times should be placed in folders named `expXXXXXX.XX`, where `XXXXXX.XX` is that captures exposure time in $\mu s$. Each image should be saved as `XXXX.npy`, where `XXXX` denotes the image number within the exposure. The black image, `black_img.npy` should also be saved within each folder, which is used for black calibration. The set of images captured at the camera's saturation limit should be saved in a folder named `saturation`, so it can be easily distinguished. The contents of this folder follow the same structure as before.

### Results
Our extensive noise modeling is able to closely capture the true noise distributions of the T6 on a per-pixel level, as evidenced by the figures below.

<p align="center">
  <img src=docs/images/comp.png>
</p>

<p align="center">
  <img src=docs/images/hist.png>
</p>

## Simulating the T6

### Example 1: all white mask
<p align="center">
  <img src=docs/images/allwhite.png>
</p>

### Example 2: all black mask
<p align="center">
  <img src=docs/images/allblack.png>
</p>

### Example 3: merge mask
<p align="center">
  <img src=docs/images/merge.png>
</p>

### Example 4: cat mask
<p align="center">
  <img src=docs/images/cat.png>
</p>

## References
Sam Hasinoff's image formation model:
> S. W. Hasinoff *et al.*, "Noise-Optimal Capture for High Dynamic Range Photography," *2010 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

Kristina Monakhova's physics-based noise model:
> K. Monakhova *et al.*, "Dancing Under the Stars: Video Denoising in Starlight," *2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*.

More information about the T6:
> R. Gulve *et al.*, "39 000-Subexposures/s Dual-ADC CMOS Image Sensor With Dual-Tap Coded-Exposure Pixels for Single-Shot HDR and 3-D Computational Imaging," *2022 IEEE Journal of Solid-State Circuits (JSSC)*.
