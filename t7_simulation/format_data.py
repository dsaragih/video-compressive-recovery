import os
import glob
import numpy as np
import argparse
import matplotlib.pyplot as plt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--data_root",
            type=str,
            help="the root direrctory of the data",
            required=True
    )
    args = parser.parse_args()

    left_raw = os.path.join(args.data_root, "left_raw_new")
    left_data = os.path.join(args.data_root, "left_data_new")
    if not os.path.exists(left_data):
        os.makedirs(left_data)

    right_raw = os.path.join(args.data_root, "right_raw_new")
    right_data = os.path.join(args.data_root, "right_data_new")
    if not os.path.exists(right_data):
        os.makedirs(right_data)


    z = 80 # Median filter radius (not diameter).


    # paths = sorted(glob.glob(left_raw + "/exp*"))
    # for i in range(len(paths)):
    #     print(f"left {i:03d}")

    #     files = sorted(glob.glob(paths[i] + "/*.npy"))

    #     imgs = []
    #     for img in files:
    #         img = np.load(img)[:,640:] * 255
    #         imgs.append(img.astype(np.uint16))
    #     imgs = np.array(imgs)
    #     imgs = np.clip((imgs[-1])[None,:,:] - imgs[:-1,:,:], 0, None)
    #     imgs = imgs.reshape(*imgs.shape, 1).astype(np.float32)

    #     mean_img = np.mean(imgs, axis=0)
    #     clean_img = np.zeros_like(mean_img)
    #     for r in range(clean_img.shape[0]):
    #         for c in range(clean_img.shape[1]):
    #             clean_img[r,c] = np.median(mean_img[max(0,r-z):min(clean_img.shape[0],r+z),max(0,c-z):min(clean_img.shape[1],c+z)])

    #     exp_dir = os.path.join(left_data, f"{i:03d}")
    #     if not os.path.exists(exp_dir):
    #         os.makedirs(exp_dir)

    #     np.save(os.path.join(exp_dir, "clean.npy"), clean_img)
    #     for j in range(imgs.shape[0]):
    #         np.save(os.path.join(exp_dir, f"noisy{j:04d}.npy"), imgs[j])

    #     exp = np.array([float(paths[i][-9:])])
    #     np.save(os.path.join(exp_dir, "exp.npy"), exp)

    fwc_dir = os.path.join(left_raw, "saturated")
    files = sorted(glob.glob(fwc_dir + "/*.npy"))

    imgs = []
    for img in files:
        img = np.load(img)[:,:640] * 255
        imgs.append(img.astype(np.uint16))
    imgs = np.array(imgs)
    # Show imgs[-1] 
    plt.imshow(imgs[-1], cmap='gray', vmin=0, vmax=2**12-1)
    plt.savefig("imgs[-1].png")

    imgs = np.clip((imgs[-1])[None,:,:] - imgs[:-1,:,:], 0, None)
    imgs = imgs.reshape(*imgs.shape, 1).astype(np.float32)

    fwc = np.mean(imgs, axis=0)
    np.save(os.path.join(left_data, "fwc.npy"), fwc)


    # paths = sorted(glob.glob(right_raw + "/exp*"))
    # for i in range(len(paths)):
    #     print(f"right {i:03d}")

    #     files = sorted(glob.glob(paths[i] + "/*.npy"))

    #     imgs = []
    #     for img in files:
    #         img = np.load(img)[:,:640] * 255
    #         imgs.append(img.astype(np.uint16))
    #     imgs = np.array(imgs)
    #     imgs = np.clip((imgs[-1])[None,:,:] - imgs[:-1,:,:], 0, None)
    #     imgs = imgs.reshape(*imgs.shape, 1).astype(np.float32)

    #     mean_img = np.mean(imgs, axis=0)
    #     clean_img = np.zeros_like(mean_img)
    #     for r in range(clean_img.shape[0]):
    #         for c in range(clean_img.shape[1]):
    #             clean_img[r,c] = np.median(mean_img[max(0,r-z):min(clean_img.shape[0],r+z),max(0,c-z):min(clean_img.shape[1],c+z)])

    #     exp_dir = os.path.join(right_data, f"{i:03d}")
    #     if not os.path.exists(exp_dir):
    #         os.makedirs(exp_dir)

    #     np.save(os.path.join(exp_dir, "clean.npy"), clean_img)
    #     for j in range(imgs.shape[0]):
    #         np.save(os.path.join(exp_dir, f"noisy{j:04d}.npy"), imgs[j])

    #     exp = np.array([float(paths[i][-9:])])
    #     np.save(os.path.join(exp_dir, "exp.npy"), exp)

    fwc_dir = os.path.join(right_raw, "saturated")
    files = sorted(glob.glob(fwc_dir + "/*.npy"))
    # Save imgs[-1] figure

    imgs = []
    for img in files:
        img = np.load(img)[:,640:] * 255
        imgs.append(img.astype(np.uint16))
    imgs = np.array(imgs)
    plt.imshow(imgs[-1], cmap='gray', vmin=0, vmax=2**12-1)
    plt.savefig("imgs_right[-1].png")
    imgs = np.clip((imgs[-1])[None,:,:] - imgs[:-1,:,:], 0, None)
    imgs = imgs.reshape(*imgs.shape, 1).astype(np.float32)

    fwc = np.mean(imgs, axis=0)
    np.save(os.path.join(right_data, "fwc.npy"), fwc)
