import numpy as np
import os
from PIL import Image
import glob

Image.MAX_IMAGE_PIXELS = 1000000000

# Set the path
path = ""
nnodes = 256

# Load the large heatmap
large_heatmap_prob = np.load(f"{path}figures_{nnodes}_updated/FCAT2APPK_heatmap_prob.npy")

# Open the original large image
large_image_path = f"{path}Palm/FCAT2APPK_new.jpg"
large_image = Image.open(large_image_path)

# Define sliding window parameters
window_size = (100, 100)
stride = 50
high_threshold = 5000 # Set this to the desired value
low_threshold = 200 # Set this to the desired value

# Create directories for high and low probability patches
high_prob_dir = os.path.join(path, "Palm/Large/high")
low_prob_dir = os.path.join(path, "Palm/Large/low")

# Create the directories if they do not exist
os.makedirs(high_prob_dir, exist_ok=True)
os.makedirs(low_prob_dir, exist_ok=True)

# Delete all the images in the high and low directories
for file_path in glob.glob(os.path.join(high_prob_dir, '*.jpg')):
    os.remove(file_path)

for file_path in glob.glob(os.path.join(low_prob_dir, '*.jpg')):
    os.remove(file_path)

# Slide the window over the large heatmap
for i in range(0, large_heatmap_prob.shape[0] - window_size[0], stride):
    for j in range(0, large_heatmap_prob.shape[1] - window_size[1], stride):
        # Crop the patch from the heatmap
        patch_prob = large_heatmap_prob[i:i + window_size[0], j:j + window_size[1]]

        # Compute the sum of probabilities within the patch
        prob_sum = np.sum(patch_prob)

        # Crop the corresponding patch from the original image
        patch_image = large_image.crop((j, i, j + window_size[1], i + window_size[0]))

        # Convert the patch to a numpy array
        patch_array = np.array(patch_image)

        # Determine whether to classify as 'high' and save the patch from the original image
        if prob_sum > high_threshold:
            save_path = os.path.join(high_prob_dir, f"patch_{i}_{j}.jpg")
            patch_image.save(save_path)
        elif prob_sum < low_threshold:
            # Check if the patch contains more than 30% white or black pixels
            white_pixels = np.sum(patch_array > 200)  # No division by 3 needed
            black_pixels = np.sum(patch_array <= 50)  # No division by 3 needed
            total_pixels = window_size[0] * window_size[1] * 3  # Multiply by 3 for RGB channels

            if (white_pixels + black_pixels) / total_pixels < 0.3:
                save_path = os.path.join(low_prob_dir, f"patch_{i}_{j}.jpg")
                patch_image.save(save_path)

print("Patches have been saved!")
