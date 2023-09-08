from PIL import Image, ImageOps
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
nnodes = 64
save_path = os.path.join(path, f"figures_{nnodes}_large_test/") 

# Predicts the large images
stride = 10
patch_size = (100, 100)
small_image_size = (5000, 5000)
Image.MAX_IMAGE_PIXELS = 1000000000
small_image = Image.open(f"{path}Palm/FCAT2APPK_new.jpg")
small_image = ImageOps.expand(small_image, (0, 0, 5, 0), fill=0)
small_heatmap_array = []
small_heatmap_array_prob = []

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
model.load_state_dict(torch.load(f'{path}figures_{nnodes}_large/resnet18_model.pth'))
model = model.to(device)
model = model.eval()

accumulator = np.zeros((small_image.height // stride, small_image.width // stride))
counter = np.ones((small_image.height // stride, small_image.width // stride))

# Iterate over the small image to get the patches
for x in range(0, small_image.width, stride):
    for y in range(0, small_image.height, stride):
        # Get a patch from the small image
        patch = small_image.crop((x, y, x + patch_size[0], y + patch_size[1]))

        # Convert patch to numpy array
        patch_array = np.array(patch)

        # Check if patch contains 25% or more black or white pixels and label it as 'non palm'
        white_pixels = np.sum(patch_array > 200) / 3  # divide by 3 to account for RGB channels
        black_pixels = np.sum(patch_array <= 50) / 3
        total_pixels = patch_size[0] * patch_size[1]

        if (white_pixels + black_pixels) / total_pixels >= 0.25:
            small_heatmap_array.append(0)  # non palm
            small_heatmap_array_prob.append(0)
            continue

        # If not 'non palm', extract features using ResNet18 and predict
        patch = patch.resize((100, 100))
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

# Repeat average_predictions 10 times along x and y axes
average_predictions_repeated = np.repeat(np.repeat(average_predictions, 10, axis=0), 10, axis=1)

# Pad average_predictions_cropped to match small_image dimensions
rows_to_pad = small_image.height - average_predictions_repeated.shape[0]
cols_to_pad = small_image.width - average_predictions_repeated.shape[1]
average_predictions_padded = np.pad(average_predictions_repeated, ((0, rows_to_pad), (0, cols_to_pad)), 'constant', constant_values=0)

# Print dimensions for debugging
print(f"Dimensions after repeating, cropping and padding: {average_predictions_padded.shape}")

# Convert to float32 before normalize
average_predictions_padded = average_predictions_padded.astype(np.float32)

# Convert the probability heatmap to a mask
mask_prob = (average_predictions_padded * 255).astype(np.uint8)

# Convert PIL Image to numpy array
small_image_np = np.array(small_image)

# Print dimensions for debugging
print(f"Dimensions of small_image: {small_image_np.shape}")
print(f"Dimensions of mask_prob before color conversion: {mask_prob.shape}")

# Convert the grayscale mask_prob to BGR
mask_prob_bgr = cv2.cvtColor(mask_prob, cv2.COLOR_GRAY2BGR)

# Print dimensions after color conversion for debugging
print(f"Dimensions of mask_prob after color conversion: {mask_prob_bgr.shape}")

# Make sure both arrays have the same dimensions before calling cv2.addWeighted
if small_image_np.shape == mask_prob_bgr.shape:
    highlighted_image_prob = cv2.addWeighted(small_image_np, 0.5, mask_prob_bgr, 0.5, 0)
else:
    print("The dimensions do not match. Cannot perform addWeighted.")

# Save the probability highlighted image
cv2.imwrite(f"{save_path}FCAT2APPK_highlighted_prob.jpg", highlighted_image_prob)
