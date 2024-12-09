import numpy as np
import cv2
from utils import _write_as_png, _generate_test_image

import os
import argparse
from pathlib import Path

def reshuffle_mosaic2vid(image, K, bucket=0):
    """
    Vectorized conversion of a mosaic image into K^2 low-resolution frames.
    
    Args:
    - image (numpy array): Input mosaic image of shape (H, W).
    - K (int): Size of each KxK tile.
    
    Returns:
    - frames (numpy array): Reshuffled frames of shape (K^2, H//K, W//K).
    """

    H, W = image.shape
    
    # Compute the padding required to make H and W divisible by K
    pad_h = (K - H % K) % K
    pad_w = (K - W % K) % K
    
    # Pad the image with zeros
    padded_image = np.pad(image, ((0, pad_h), (0, pad_w)), mode='constant', constant_values=0)
    
    # Get the new padded dimensions
    new_H, new_W = padded_image.shape
    
    # Reshape the image to extract KxK tiles
    padded_image = padded_image[::-1, :]  # Flip the image vertically
    # (H//K, K, W//K, K): reshapes into submatrices where each is KxK
    reshaped_image = padded_image.reshape(new_H // K, K, new_W // K, K)
    
    # Transpose to get the KxK tile dimensions aligned for reshuffling
    # Now shape is (H//K, W//K, K, K)
    transposed_image = reshaped_image.transpose(0, 2, 1, 3)
    
    # Reshape to combine the KxK tiles into K^2 separate frames
    # (H//K, W//K, K^2): each KxK tile is now flattened into K^2 and frames are across the grid
    flattened_image = transposed_image.reshape(new_H // K, new_W // K, K * K)
    
    # Transpose to get the final frame structure
    # (K^2, H//K, W//K): now we have K^2 frames of size H//K x W//K
    frames = flattened_image.transpose(2, 0, 1)
    frames = frames[::-1, ::-1, :] if bucket == 0 else frames[:, ::-1, :]
    
    return frames


if __name__ == "__main__":
    # Example usage
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_path',
        type=str,
        help='Path to the mosaic image',
        required=True
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        help='Path to the output folder',
        required=True
    )

    args = parser.parse_args()
    image_path = args.image_path
    output_dir = args.output_dir

    K = 4
    # # Presume 320 x 640 image
    image = cv2.imread(image_path, 0)
    image = (image.astype(float) * 16)
    # image = np.tile(_generate_test_image(K, 320), (1, 2)) * 256 * 16
    print(f"Input image shape: {image.shape}")
    # Each bucket individually
    size = image.shape[1] // 2
    image_0 = image[:, :size]
    image_1 = image[:, size:]

    # _write_as_png(f"./outputs/frame.png", image)
    _write_as_png(os.path.join(output_dir, f"{Path(image_path).stem}_left.png"), image_0)
    _write_as_png(os.path.join(output_dir, f"{Path(image_path).stem}_right.png"), image_1)

    frames_0 = reshuffle_mosaic2vid(image_0, K)
    frames_1 = reshuffle_mosaic2vid(image_1, K, 1)

    # print(f"Reshuffled frames shape: {frames_0.shape}")
    # print(f"Reshuffled frames shape: {frames.shape}")

    # # Save the frames as .npy and .png files
    # Show base images
    file_name = __file__.split("/")[-1].split(".")[0]
    for i, frame in enumerate(frames_0):
        # The output folder in inverse_solvers
        # Resize
        # frame = cv2.resize(frame, (frame.shape[1] * (K), frame.shape[0] * (K)), interpolation=cv2.INTER_LINEAR)
        _write_as_png(os.path.join(output_dir, f"{Path(image_path).stem}_left_{i:05d}.png"), frame)
        # _write_as_png(f"./outputs/{file_name}_frame_0_{i:05d}.png", frame)
    for i, frame in enumerate(frames_1):
        # The output folder in inverse_solvers
        # Resize
        # frame = cv2.resize(frame, (frame.shape[1] * (K), frame.shape[0] * (K)), interpolation=cv2.INTER_LINEAR)
        _write_as_png(os.path.join(output_dir, f"{Path(image_path).stem}_right_{i:05d}.png"), frame)
        # _write_as_png(f"./outputs/{file_name}_frame_1_{i:05d}.png", frame)
