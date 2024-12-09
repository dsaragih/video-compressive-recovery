import os
import glob
import numpy as np
import cv2
import scipy.io as sio
import argparse

def _save_image(img, img_path):
    # cv2.imwrite(os.path.join("./outputs", img_path), np.clip(img, 0, 255).astype(np.uint8))
    pass

def simulate(left_clean, right_clean, params, mask, num_burst, subframes, exp):
    _save_image(left_clean[0,0], "left_clean.png")
    left_noisy = left_clean * params["left_g"] + \
            params["left_h"]  * (left_mask.sum(axis=0) / subframes)[None,None] + \
            params["left_dark"]**2 * exp * left_mask.sum(axis=0)[None,None]
    left_noisy = np.clip(left_noisy, 0, params["left_fwc"])
    right_noisy = right_clean * params["right_g"] + \
            params["right_h"]  * (right_mask.sum(axis=0) / subframes)[None,None] + \
            params["right_dark"]**2 * exp * right_mask.sum(axis=0)[None,None]
    right_noisy = np.clip(right_noisy, 0, params["right_fwc"])

    _save_image(left_noisy[0,0], "left_noisy.png")

    left_shot = np.random.normal(size=(num_burst, 1, H, W)) * params["left_shot"] * np.sqrt(left_clean) * params["left_g"]
    left_shot[left_noisy >= params["left_fwc"]] = 0
    left_noisy += left_shot
    right_shot = np.random.normal(size=(num_burst, 1, H, W)) * params["right_shot"] * np.sqrt(right_clean) * params["right_g"]
    right_shot[right_noisy >= params["right_fwc"]] = 0
    right_noisy += right_shot

    _save_image(left_noisy[0,0], "left_noisy_shot.png")

    left_read = np.random.normal(size=(num_burst, 1, H, W)) * params["left_read"]
    left_noisy += left_read
    right_read = np.random.normal(size=(num_burst, 1, H, W)) * params["right_read"]
    right_noisy += right_read

    _save_image(left_noisy[0,0], "left_noisy_read.png")

    left_row = np.random.normal(size=(num_burst, 1, H, 1)) * params["left_row"]
    left_noisy += left_row
    right_row = np.random.normal(size=(num_burst, 1, H, 1)) * params["right_row"]
    right_noisy += right_row

    _save_image(left_noisy[0,0], "left_noisy_row.png")

    left_rowt = np.random.normal(size=(1, 1, H, 1)) * params["left_rowt"]
    left_noisy += left_rowt
    right_rowt = np.random.normal(size=(1, 1, H, 1)) * params["right_rowt"]
    right_noisy += right_rowt

    _save_image(left_noisy[0,0], "left_noisy_rowt.png")

    left_quant = np.random.uniform(size=(num_burst, 1, H, W)) * params["left_quant"]
    left_noisy += left_quant
    right_quant = np.random.uniform(size=(num_burst, 1, H, W)) * params["right_quant"]
    right_noisy += right_quant

    _save_image(left_noisy[0,0], "left_noisy_quant.png")

    left_dark = np.random.normal(size=(num_burst, 1, H, W)) * params["left_dark"] * np.sqrt(exp * left_mask.sum(axis=0)[None,None])
    left_noisy += left_dark
    right_dark = np.random.normal(size=(num_burst, 1, H, W)) * params["right_dark"] * np.sqrt(exp * right_mask.sum(axis=0)[None,None])
    right_noisy += right_dark

    return np.concatenate((left_noisy, right_noisy), axis=3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--params",
            type=str,
            help="the path to the parameters .mat file",
            required=True
    )
    parser.add_argument(
            "--mask",
            type=str,
            help="the path to the mask .bmp file",
            required=True
    )
    parser.add_argument(
            "--input_imgs",
            type=str,
            help="the path to the input image(s) (for multiple images, use ? [e.g. images/?.png])",
            required=True
    )
    parser.add_argument(
            "--output_fname",
            type=str,
            help="the path to the output file",
            required=True
    )
    parser.add_argument(
            "--mode",
            type=str,
            help="options: " + \
                    "single_in (uses a single input image), " + \
                    "multi_in_single_out (uses multiple input images, switching at each subframe), " + \
                    "multi_in_multi_out (uses multiple input images, switching at each capture)",
            default="single_in"
    )
    parser.add_argument(
            "--num_burst",
            type=int,
            help="number of burst captures (only used for single_in and multi_in_single_out modes)",
            default=1
    )
    parser.add_argument(
            "--subframes",
            type=int,
            help="number of subframes per capture (only used for single_input and multi_in_multi_out modes)",
            default=1
    )
    parser.add_argument(
            "--exp",
            type=float,
            help="exposure time [us] of each subframe (>= 26.21)",
            default=78.01
    )
    parser.add_argument(
            "--scale",
            type=float,
            help="linear scale that can be used to scale input images prior to simulation",
            default=1.0
    )
    parser.add_argument(
        "--cam_type",
        type=str,
        choices=["t6", "t7"],
        default="t7"
    )
    args = parser.parse_args()

    params = sio.loadmat(args.params)
    global H
    global W
    if args.cam_type == "t6":
        H, W = 320, 320
    elif args.cam_type == "t7":
        H, W = 480, 640
    else:
        raise ValueError("Invalid camera type")
    print("H:", H, "W:", W)

    mask = cv2.imread(args.mask, 0)
    assert mask.shape[0] % H == 0 and mask.shape[1] >= W
    mask = mask[::-1,:W]

    img_paths = sorted(glob.glob(args.input_imgs.replace('?', '*')))
    print("img_paths:", img_paths)
    assert len(img_paths) > 0
    input_imgs = []
    for i in range(len(img_paths)):
        # print("Reading image", img_paths[i])
        img = cv2.imread(img_paths[i], 0)
        # need to flip dimensions for cv2
        img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)
        img = (img.astype(float))
        input_imgs.append(img)
    input_imgs = np.stack(input_imgs)

    output_fname = args.output_fname
#     output_dir = args.output_dir
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)

    mode = args.mode
    assert mode in ["single_in", "multi_in_single_out", "multi_in_multi_out"]

    num_burst = args.num_burst
    assert num_burst > 0

    subframes = args.subframes
    assert subframes > 0

    exp = args.exp
    assert exp >= 26.21

    scale = args.scale
    assert scale >= 0
    input_imgs = input_imgs * scale

    if mode == "single_in":
        mask_re = np.zeros((subframes * H, W))
        mask_re[:min(mask.shape[0], mask_re.shape[0])] = mask[:min(mask.shape[0], mask_re.shape[0])]
        mask_re = mask_re.reshape(subframes, H, W)
        left_mask = (mask_re > 0).astype(int)
        right_mask = (mask_re == 0).astype(int)

        input_imgs = np.repeat(input_imgs[0][None], subframes, axis=0)

        left_clean = np.repeat((input_imgs * left_mask).sum(axis=0)[None,None], num_burst, axis=0)
        right_clean = np.repeat((input_imgs * right_mask).sum(axis=0)[None,None], num_burst, axis=0)

    elif mode == "multi_in_single_out":
        subframes = input_imgs.shape[0]

        mask_re = np.zeros((subframes * H, W))
        mask_re[:min(mask.shape[0], mask_re.shape[0])] = mask[:min(mask.shape[0], mask_re.shape[0])]
        mask_re = mask_re.reshape(subframes, H, W)
        left_mask = (mask_re > 0).astype(int)
        right_mask = (mask_re == 0).astype(int)

        left_clean = np.repeat((input_imgs * left_mask).sum(axis=0)[None,None], num_burst, axis=0)
        right_clean = np.repeat((input_imgs * right_mask).sum(axis=0)[None,None], num_burst, axis=0)

    else:
        num_burst = input_imgs.shape[0]

        mask_re = np.zeros((subframes * H, W))
        mask_re[:min(mask.shape[0], mask_re.shape[0])] = mask[:min(mask.shape[0], mask_re.shape[0])]
        mask_re = mask_re.reshape(subframes, H, W)
        left_mask = (mask_re > 0).astype(int)
        right_mask = (mask_re == 0).astype(int)

        input_imgs = np.repeat(input_imgs[None], subframes, axis=0)

        left_clean = (input_imgs * left_mask[:,None]).sum(axis=0)[:,None]
        right_clean = (input_imgs * right_mask[:,None]).sum(axis=0)[:,None]

    output_imgs = simulate(left_clean, right_clean, params, mask, num_burst, subframes, exp)

    mask_name = os.path.basename(args.mask).split(".")[0]
    print("output_imgs.shape:", output_imgs.shape)
    for i in range(output_imgs.shape[0]):
        np.save(f"{output_fname}_{i}.npy", output_imgs[i,0])
        cv2.imwrite(f"{output_fname}_{i}.png", np.clip(output_imgs[i,0], 0, 255).astype(np.uint8))

        # np.save(os.path.join(output_dir, f"{mask_name}_{i:05d}.npy"), output_imgs[i,0])
        # cv2.imwrite(os.path.join(output_dir, f"{mask_name}_{i:05d}.png"), np.clip(output_imgs[i,0] / 16, 0, 255).astype(np.uint8))
