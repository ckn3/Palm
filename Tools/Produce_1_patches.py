import cv2
import numpy as np
from PIL import Image

# Initialize a counter for the patches
patch_counter = 0
patch_stop = 30

# Function to check if a pixel is isolated
def is_isolated(i, j, heatmap, threshold=0):
    rows, cols = heatmap.shape
    count = 0
    for x in range(max(0, i-1), min(rows, i+2)):
        for y in range(max(0, j-1), min(cols, j+2)):
            if x != i or y != j:  # Exclude the pixel itself
                if heatmap[x, y] == 1:
                    count += 1
    return count <= threshold

# Iterate over all pairs of images and heatmaps
for k in range(patch_counter, patch_stop):
    # Load the heatmap
    small_heatmap = np.load(f'Palm/small_images_new/small_image_{k}_heatmap.npy')

    # Load the image
    small_image = Image.open(f'Palm/small_images_new/small_image_{k}.jpg')

    # # Flip the heatmap horizontally and rotate 90 degrees anticlockwise
    # small_heatmap = np.flip(small_heatmap, 1)
    # small_heatmap = np.rot90(small_heatmap, 1)

    # Convert the image to a NumPy array for processing
    small_image_np = np.array(small_image)

    # Convert the PIL Image back to an OpenCV image (BGR to RGB)
    small_image_cv = cv2.cvtColor(small_image_np, cv2.COLOR_RGB2BGR)

    # Create a copy of the image to draw on
    image_with_boxes = small_image_cv.copy()

    # Iterate over the heatmap and image to crop patches and draw rectangles
    for i in range(small_heatmap.shape[0]):
        for j in range(small_heatmap.shape[1]):
            # If the heatmap value at this location is 1, crop the corresponding patch from the image
            if small_heatmap[i, j] == 1 and is_isolated(i, j, small_heatmap):
                # Calculate the top left corner of the patch in the image
                top_left_x = j * 40
                top_left_y = i * 40

                # Draw a rectangle on the image
                image_with_boxes = cv2.rectangle(image_with_boxes, (top_left_x, top_left_y), (top_left_x+40, top_left_y+40), (0, 0, 255), 2)

                # Crop the patch from the image
                patch = small_image_np[top_left_y:top_left_y+40, top_left_x:top_left_x+40, :]

                # Convert the patch back to an image
                patch_img = Image.fromarray(patch)

                # Save the patch
                patch_img.save(f'Palm/patches/patch_{k}_{patch_counter}.jpg')

                patch_counter += 1

    # Save the image with boxes
    cv2.imwrite(f'Palm/patches/highlighted/highlighted_image_{k}.jpg', image_with_boxes)