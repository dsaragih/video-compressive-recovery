import numpy as np
import cv2
import argparse
from PIL import Image

def save_as_1bit_bmp(img_array, filename):
    # Convert the numpy array to an image
    img = Image.fromarray(img_array, mode='L')
    # Convert to 1-bit monochrome
    img = img.convert('1')
    # Save as BMP
    img.save(filename, "BMP")



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--subframes',
        type=int,
        help='Number of subframes. MUST be a square number',
        required=False,
        default=4
    )
    parser.add_argument(
        '--frame_width',
        type=int,
        help='Width of the frame. For T7, do not change this value from 1024',
        required=False,
        default=1024
    )

    parser.add_argument(
        '--frame_height',
        type=int,
        help='Height of the frame',
        required=False,
        default=480
    )

    return parser.parse_args()

def get_pattern(K):
    # Initialize an empty list to hold the patterns
    patterns = []
    
    # Generate K^2 patterns
    for i in range(K):
        for j in range(K):
            # Create a K x K matrix filled with zeros
            pattern = np.zeros((K, K), dtype=int)
            # Set the (i, j)-th element to 1
            pattern[i, j] = 1
            # Flip pattern along the horizontal axis
            pattern = np.flip(pattern, axis=0)
            # Append this pattern to the list
            patterns.append(pattern)
    
    # Convert the list of patterns to a 3D numpy array of shape (K^2, K, K)
    patterns_array = np.array(patterns)
    
    return patterns_array


if __name__ == '__main__':
    args = parse_args()

    # subframes must be a square number
    tile_size = int(np.sqrt(args.subframes))
    img_height = args.frame_height
    img_width = args.frame_width

    patterns = get_pattern(tile_size)

    mask = np.zeros((args.subframes, img_height, img_width), dtype=np.uint8)
    tile_multiplier_height = img_height // tile_size + 1 # we cut off the extra pixels later
    tile_multiplier_width = img_width // tile_size + 1

    for i in range(args.subframes):
        pattern = patterns[i]
        mask[i] = np.tile(pattern, (tile_multiplier_height, tile_multiplier_width))[:img_height, :img_width]
    
    mask = mask.reshape(args.subframes * img_height, img_width)
    mask *= 255

    # Save the mask as 1-bit BMP
    save_as_1bit_bmp(mask, f'masks/trial_coded_exposure_{tile_size}x{tile_size}.bmp')
