import numpy as np
import cv2
from utils import _write_as_png, _generate_test_image
from scipy.interpolate import griddata


def pinv_mosaic2vid(image, mask_matrix):
    """
    Compute the pseudo-inverse of the given masking matrix using the Moore-Penrose inverse.
    
    Args:
    - image (numpy array): Input mosaic image of shape (H, W).
    - mask_matrix (numpy array): Masking matrix of shape (K, H, W).
    
    Returns:
    - frames (numpy array): Frames of shape (K, H, W).
    """
    K, H, W = mask_matrix.shape
    
    # Reshape image and mask_matrix to apply pinv across all pixels at once
    image_flat = image.reshape(H*W, 1)  # Shape (H*W, 1)
    mask_matrix_flat = mask_matrix.reshape(K, H * W)  # Shape (K, H*W)

    # Compute the pseudo-inverse for each pixel position (broadcasting along the columns)
    pseudo_inverse = np.linalg.pinv(mask_matrix_flat.T.reshape(H * W, 1, K))  # Shape (H * W, K, 1)

    print(f"Pseudo-inverse shape: {pseudo_inverse.shape}")
    
    # Multiply pseudo-inverse with the image values to reconstruct the frames
    frames_flat = pseudo_inverse.reshape(H*W, K) * image_flat  # Shape (H*W, K)
    
    # Reshape the frames back to (K, H, W)
    frames = frames_flat.T.reshape(K, H, W)
    
    return frames

def forloop_pinv_mosaic2vid(image, mask_matrix):
    
    frames = np.zeros(mask_matrix.shape)
    H, W = image.shape
    for i in range(H):
        for j in range(W):
            mask = mask_matrix[:, i, j] # Shape is (K,)

            pixel_frames = np.linalg.pinv(mask.reshape(1, -1)) * image[i, j]
            frames[:, i, j] = pixel_frames.flatten()

    return frames

def _get_masks(mask):
    # Get the masks for each bucket
    # mask is a 3D array of shape (K, H, W)
    # put 1 where mask is > 0, 0 otherwise
    # for bucket 1 mask, put 1 where mask is == 0, 0 otherwise
    left_mask = (mask > 0).astype(int)
    right_mask = (mask == 0).astype(int)

    return left_mask, right_mask

def interpolate_pinv(pinv):
    # Interpolate the frames using the pinv
    K, H, W = pinv.shape
    frames = np.zeros((K, H, W))

    for k in range(K):
        # Extract the channel where the mask is non-zero
        channel = pinv[k]

        # Get the coordinates of non-zero values in the mask
        x, y = np.where(channel != 0)
        values = channel[x, y]  # Non-zero values in the channel

        # Generate a grid of all points (H, W)
        grid_x, grid_y = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')

        # Perform interpolation using griddata
        interpolated = griddata((x, y), values, (grid_x, grid_y), method='linear', fill_value=0)

        # Store the interpolated frame
        frames[k] = interpolated

    return frames

if __name__ == "__main__":
    # Example usage
    image_path = "/home/daniel/t6_simulation/outputs/coded_exposure_2x2_00000.png"
    mask_path = "/home/daniel/t6_simulation/masks/coded_exposure_2x2.bmp"
    K = 2
    # # Presume 320 x 640 image
    image = cv2.imread(image_path, 0)
    image = (image.astype(float) * 16)
    # image = np.tile(_generate_test_image(K, 320), (1, 2)) * 256 * 16
    print(f"Input image shape: {image.shape}")
    # Each bucket individually
    size = image.shape[0]
    image_0 = image[:, :size]
    image_1 = image[:, size:]

    mask = cv2.imread(mask_path, 0)
    assert mask.shape[0] % 320 == 0 and mask.shape[1] >= 320
    mask = mask[::-1,:320].reshape(mask.shape[0] // 320, 320, 320)
    left_mask, right_mask = _get_masks(mask)

    _write_as_png(f"./outputs/frame.png", image)
    _write_as_png(f"./outputs/frame_0.png", image_0)
    _write_as_png(f"./outputs/frame_1.png", image_1)

    frames_0 = pinv_mosaic2vid(image_0, left_mask)
    frames_0 = interpolate_pinv(frames_0)
    frames_1 = pinv_mosaic2vid(image_1, right_mask)
    frames_1 = interpolate_pinv(frames_1)

    print(f"Reshuffled frames shape: {frames_0.shape}")
    # print(f"Reshuffled frames shape: {frames.shape}")

    # # Save the frames as .npy and .png files
    filename = __file__.split("/")[-1].split(".")[0]
    # Show base images
    for i, frame in enumerate(frames_0):
        # The output folder in inverse_solvers
        _write_as_png(f"./outputs/pinv/{filename}_frame_0_{i:05d}.png", frame)
    for i, frame in enumerate(frames_1):
        # The output folder in inverse_solvers
        _write_as_png(f"./outputs/pinv/{filename}_frame_1_{i:05d}.png", frame)
