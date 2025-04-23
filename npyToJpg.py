import numpy as np
from PIL import Image

# Load the .npy file
npy_file = 'datasets/vehicle-edge-671-4ctg/ei-vehicle-detection-bd-v5-image-X_training.npy'
image_array = np.load(npy_file)

# Optionally normalize if the values are not in range 0-255
# If the image has float values, normalize them to 0-255
for x in image_array:
    if x.dtype != np.uint8:
        x = (255 * (x - np.min(x)) / (np.max(x) - np.min(x))).astype(np.uint8)

    # If the image has only one channel (grayscale), ensure the array is 2D
    if len(x.shape) == 2:
        img = Image.fromarray(x, mode='L')  # 'L' mode is for grayscale
    elif len(x.shape) == 3 and x.shape[2] == 3:
        img = Image.fromarray(x, mode='RGB')  # 'RGB' mode for color images
    else:
        raise ValueError("Unsupported image format!")

    # Save as a .jpg file
    output_file = 'output_image.jpg'
    img.save(output_file)

    print(f"Image saved as {output_file}")
