from PIL import Image

Image.MAX_IMAGE_PIXELS = None
# Open the original large image to get its dimensions
original_large_image_path = "Palm/FCAT2APPK.jpg"
original_large_image = Image.open(original_large_image_path)
original_width, original_height = original_large_image.size

img_counter = 30
patch_size = (40, 40)
small_image_size = (5000, 5000)
Image.MAX_IMAGE_PIXELS = 1000000000

# Create a new image with a slightly larger size
new_image_width, new_image_height = 30000, 25000
large_image = Image.new('RGB', (new_image_width, new_image_height))

# Specify the number of small images in each dimension
num_images_width = 6  # Change this based on your specific arrangement
num_images_height = 5  # Change this based on your specific arrangement

# Place each small image onto the large image canvas in the right order
for i in range(num_images_width):
    for j in range(num_images_height):
        if i * num_images_height + j >= img_counter:  # Only combine 30 images
            break
        small_image = Image.open(f"Palm/small_images_new/small_image_{i * num_images_height + j}.jpg")
        large_image.paste(small_image, (i * small_image_size[0], j * small_image_size[1]))

# Crop the combined image to match the original large image size
large_image = large_image.crop((0, 0, original_width, original_height))

# Save the resulting large image
large_image.save("FCAT2APPK_new.jpg")
