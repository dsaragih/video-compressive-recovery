import numpy as np
import cv2
from utils import _write_as_png, _generate_test_image
from scipy.interpolate import griddata

def interpolate_mosaic2vid(image, mask_matrix, show=True):
    """
    Parameters:
    - image (numpy array): Input mosaic image of shape (H, W).
    - mask_matrix (numpy array): Masking matrix of shape (K, H, W).

    Returns:
    - frames (numpy array): Frames of shape (K, H, W).
    """

    K, H, W = mask_matrix.shape
    frames = np.zeros((K, H, W))

    for k in range(K):
        # Extract the channel where the mask is non-zero
        channel = image * mask_matrix[k]

        if show:
            _write_as_png(f"./outputs/fan_5fps/channel_{k}.png", channel)
        
        # Get the coordinates of non-zero values in the mask
        x, y = np.where(mask_matrix[k] != 0)
        values = channel[x, y]  # Non-zero values in the channel
        
        # Generate a grid of all points (H, W)
        grid_x, grid_y = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        
        # Perform interpolation using griddata
        interpolated = griddata((x, y), values, (grid_x, grid_y), method='linear', fill_value=0)
        
        # Store the interpolated frame
        frames[k] = interpolated

    return frames

def _show_masks(path, mask, n_tiles=1):
    # Mask of (K, H, W)

    # crop the mask to be (K, H-crop, W-crop)
    tile_size = int(mask.shape[0] ** 0.5)
    crop = tile_size * n_tiles
    mask = mask[:, :crop, :crop]

    # flatten
    for i, m in enumerate(mask):
        # Enlarge me
        m = cv2.resize(m, (m.shape[1] * 32, m.shape[0] * 32), interpolation=cv2.INTER_NEAREST)
        _write_as_png(f"{path[:-4]}_{i}.png", m * 255 * 16)

if __name__ == "__main__":
    # Example usage
    image_path = "./../outputs/fan_4x4_video.png"
    mask_path = "./../masks/t7_coded_exposure_4x4.bmp"
    K = 4
    H, W = 480, 640
    # # Presume 320 x 640 image
    image = cv2.imread(image_path, 0)
    image = (image.astype(float) * 16)
    # image = np.tile(_generate_test_image(K, 320), (1, 2)) * 256 * 16
    print(f"Input image shape: {image.shape}")
    # Each bucket individually
    image_0 = image[:, :W]
    image_1 = image[:, W:]

    mask = cv2.imread(mask_path, 0)
    assert mask.shape[0] % H == 0 and mask.shape[1] >= W
    mask = (mask > 0).astype(int)
    left_mask = mask[::-1,:W].reshape(mask.shape[0] // H, H, W)
    right_mask = mask[::-1,:W].reshape(mask.shape[0] // H, H, W)

    _write_as_png(f"./outputs/frame.png", image)
    _write_as_png(f"./outputs/frame_0.png", image_0)
    _write_as_png(f"./outputs/frame_1.png", image_1)

    # _show_masks(f"./outputs/left_mask.png", left_mask)
    # _show_masks(f"./outputs/right_mask.png", right_mask)

    frames_0 = interpolate_mosaic2vid(image_0, left_mask)
    frames_1 = interpolate_mosaic2vid(image_1, right_mask)

    print(f"Reshuffled frames shape: {frames_0.shape}")
    # print(f"Reshuffled frames shape: {frames.shape}")

    # # Save the frames as .npy and .png files
    # Show base images
    for i, frame in enumerate(frames_0):
        # The output folder in inverse_solvers
        _write_as_png(f"./outputs/interp/frame_0_{i:05d}.png", frame)
    for i, frame in enumerate(frames_1):
        # The output folder in inverse_solvers
        _write_as_png(f"./outputs/interp/frame_1_{i:05d}.png", frame)
