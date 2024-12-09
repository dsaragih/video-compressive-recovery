import cv2
import numpy as np
import os

def _write_as_png(file_path, image):
    """
    Write the image as a PNG file.
    
    Args:
    - file_path (str): Path to the output PNG file.
    - image (numpy array): Image to write.
    """
    # Extract the directory from the path
    dir = os.path.dirname(file_path)
    
    # Check if the directory exists, if not, create it
    if not os.path.exists(dir):
        os.mkdir(dir)
    cv2.imwrite(file_path, np.clip(image / 16, 0, 255).astype(np.uint8))

def _generate_test_image(K, size):
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
                # Append this pattern to the list
                patterns.append(pattern)
        
        # Convert the list of patterns to a 3D numpy array of shape (K^2, K, K)
        patterns_array = np.array(patterns)
        
        return patterns_array
    image = get_pattern(K)
    weights = np.linspace(0, 1, K * K)
    image = (image * weights[:, None, None]).sum(axis=0)
    print(f"Generated image shape: {image.shape}")
    image = np.tile(image, (size // K, size // K))

    return image