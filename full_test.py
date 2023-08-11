from PIL import Image
import numpy as np
import os
import re
import pandas.core.common as com
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import lr_scheduler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_curve, roc_auc_score, classification_report, cohen_kappa_score, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import cv2

path = "/deac/csc/paucaGrp/student-research-team/palm_detection/"  # Change this line to set your custom path /deac/csc/paucaGrp/student-research-team/palm_detection/

nnodes = 128

# Predicts the large images
img_counter=30
patch_size = (40, 40)
small_image_size = (5000, 5000)
Image.MAX_IMAGE_PIXELS = 1000000000
large_image_path = f"{path}Palm/FCAT2APPK_new.jpg"
large_image = Image.open(large_image_path)

def initialize_model():
    model = models.resnet18(pretrained=False)

    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
                   # nn.Dropout(p=0.2),
                   nn.Linear(num_ftrs, nnodes),
                   nn.ReLU(),
                   nn.BatchNorm1d(nnodes),
                   nn.Linear(nnodes, 2),  # here, 2 is the number of classes
                   nn.LogSoftmax(dim=1)
    )

    return model

def preprocess_test(patch_img):
    # Define the transform
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return preprocess(patch_img)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = initialize_model()
model.load_state_dict(torch.load(f'{path}figures_{nnodes}_updated/resnet18_model.pth'))
model = model.to(device)
model = model.eval()

initial_image_number = 0
stride = 10

for i in range(initial_image_number, img_counter):
    small_image = Image.open(f"{path}Palm/small_images_new/small_image_{i}.jpg")
    small_heatmap_array = []
    small_heatmap_array_prob = []
    
    # Initialize the accumulator and counter arrays
    accumulator = np.zeros((small_image.height // stride, small_image.width // stride))
    counter = np.ones((small_image.height // stride, small_image.width // stride))

    # Iterate over the small image to get the patches
    for x in range(0, small_image.width, stride):
        for y in range(0, small_image.height, stride):
            # Get a patch from the small image
            patch = small_image.crop((x, y, x + patch_size[0], y + patch_size[1]))

            # Convert patch to numpy array
            patch_array = np.array(patch)

            # Check if patch contains 40% or more black or white pixels and label it as 'non palm'
            white_pixels = np.sum(patch_array > 200) / 3  # divide by 3 to account for RGB channels
            black_pixels = np.sum(patch_array <= 50) / 3
            total_pixels = patch_size[0] * patch_size[1]

            if (white_pixels + black_pixels) / total_pixels >= 0.4:
                small_heatmap_array.append(0)  # non palm
                small_heatmap_array_prob.append(0)
                continue

            # If not 'non palm', extract features using ResNet18 and predict
            patch = patch.resize((40, 40))
            patch_img = Image.fromarray(patch_array.astype('uint8'), 'RGB')

            # Preprocess the patch
            patch_t = preprocess_test(patch_img)
            patch_u = torch.unsqueeze(patch_t, 0)
            patch_u = patch_u.to(device)  # Remember to move your inputs to the same device as your model

            # Extract features
            with torch.no_grad():
                output = model(patch_u)

            # Apply softmax to get probabilities and find the class with maximum probability
            probabilities = torch.exp(output).cpu()
            
            # Add the prediction to the accumulator and increment the counter for all overlapping patches
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    if 0 <= x // stride + dx < accumulator.shape[1] and 0 <= y // stride + dy < accumulator.shape[0]:
                        accumulator[y // stride + dy, x // stride + dx] += probabilities[0][1].item()
                        counter[y // stride + dy, x // stride + dx] += 1

        # After all predictions have been made, divide the accumulator by the counter to get the average predictions
    average_predictions = accumulator / counter

    # Reshape the average_predictions to the shape of the small image
    average_predictions = average_predictions.reshape((small_image.height // stride, small_image.width // stride))
    print(np.shape(average_predictions))

    # Convert PIL Image to numpy array
    small_image_np = np.array(small_image)
    
    # Compute the scaling factors
    scale_y = small_image_np.shape[0] / average_predictions.shape[0]
    scale_x = small_image_np.shape[1] / average_predictions.shape[1]

    # Resize the heatmap
    average_predictions_resized = np.repeat(np.repeat(average_predictions, stride, axis=0), stride, axis=1)
    
    # Save the heatmap array and the corresponding figure
    np.save(f"{path}figures_{nnodes}_updated/small_image_{i}_heatmap_prob.npy", average_predictions_resized)

    # Convert to float32 before normalize
    average_predictions_resized = average_predictions_resized.astype(np.float32)

    # Convert the probability heatmap to a mask
    mask_prob = (average_predictions_resized * 255).astype(np.uint8)

    # Convert PIL Image to numpy array
    small_image = np.array(small_image)

    # Highlight the palms in the image using the probability mask
    highlighted_image_prob = cv2.addWeighted(small_image, 0.5, cv2.cvtColor(mask_prob, cv2.COLOR_GRAY2BGR), 0.5, 0)

    # Save the probability highlighted image
    cv2.imwrite(f"{path}figures_{nnodes}_updated/small_image_{i}_highlighted_prob.jpg", highlighted_image_prob)

# Create a new image with a slightly larger size
new_image_width, new_image_height = 30000, 25000
large_heatmap_prob = np.zeros((new_image_height, new_image_width))
large_image_highlighted_prob = Image.new('RGB', (new_image_width, new_image_height))

# Specify the number of small images in each dimension
num_images_width = new_image_width // small_image_size[0]
num_images_height = new_image_height // small_image_size[1]

for i in range(num_images_width):
    for j in range(num_images_height):
        # Place each small image onto the large image canvas in the right order
        small_image_highlighted_prob = Image.open(f"{path}figures_{nnodes}_updated/small_image_{i*num_images_height + j}_highlighted_prob.jpg")
        large_image_highlighted_prob.paste(small_image_highlighted_prob, (i*small_image_size[0], j*small_image_size[1]))
        # Place the small heatmap onto the large heatmap canvas
        small_heatmap_prob = np.load(f"{path}figures_{nnodes}_updated/small_image_{i*num_images_height + j}_heatmap_prob.npy")
        large_heatmap_prob[j*small_image_size[1]:(j+1)*small_image_size[1], i*small_image_size[0]:(i+1)*small_image_size[0]] = small_heatmap_prob

# Crop the combined image to the original image size
large_image_highlighted_prob = large_image_highlighted_prob.crop((0, 0, large_image.width, large_image.height))

# Crop the combined heatmap to the original image size
large_heatmap_prob = large_heatmap_prob[0:large_image.height, 0:large_image.width]

# Save the resulting large image
large_image_highlighted_prob.save(f"{path}figures_{nnodes}_updated/FCAT2APPK_highlighted_prob.jpg")

# Save the resulting large heatmap
np.save(f"{path}figures_{nnodes}_updated/FCAT2APPK_heatmap_prob.npy", large_heatmap_prob)