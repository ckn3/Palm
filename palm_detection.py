# Import necessary libraries
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

# Set the path to the workspace
path = "/deac/csc/paucaGrp/student-research-team/palm_detection/"  # Change this line to set your custom path /deac/csc/paucaGrp/student-research-team/palm_detection/
# Set the number of nodes
nnodes = 256
# Set the path to save figures
save_path = os.path.join(path, f"figures_{nnodes}_updated/")

# Unique parts of your folder paths
folders = ['Palm/non_palm',
           'Palm/patches/nonpalms',
           'Palm/patches/palms',
           'Palm/slice',
           'Palm/tree_non_palm',
           'Palm/minority']

# Prepending the path to each folder
folders = [path + folder for folder in folders]


# Prepare empty lists to store your data and labels
X = []
X4 = []
X_filenames = []  # New list to store filenames
Y = []

# Iterate over folders and files, and load your data
for folder in folders:
    for filename in os.listdir(folder):
        img = Image.open(os.path.join(folder, filename))
        X_filenames.append(filename)  # Save the filename

        # Resize the image to 40x40
        img = img.resize((40, 40))

        # Convert the image to numpy array and separate the channels
        img_array = np.array(img)
        img_rgb = img_array[:, :, :3]  # first three channels (assuming RGB)
        X.append(img_rgb)

        if img_array.shape[2] == 4:  # If the image has a fourth channel
            img_fourth = img_array[:, :, 3]  # fourth channel
            X4.append(img_fourth)

        # Set the labels
        if folder.endswith('non_palm') or folder.endswith('nonpalms'):
            # For 'non_palm' and 'non_palms' folders, the label is 'non palm'
            Y.append('non palm')
        else:
            # For the other folders, the label is the third part of the filename
            # label = re.split('_+', filename)[2]
            # Y.append(label)
            Y.append('palm')

# Convert lists to numpy arrays
X = np.array(X)
X4 = np.array(X4)
Y = np.array(Y)

# Convert 'road' to 'non palm' in Y
Y = ['non palm' if y == 'Road' else y for y in Y]

binary_task = True  # Set to False for multiclass task

# Convert all non-'non palm' labels to 'palm' in Y if the task is binary
if binary_task:
    Y = ['palm' if y != 'non palm' else y for y in Y]

# No need to reshape X as we've explicitly resized all images to 40x40
# p is simply the length of X
p = len(X)

# Get the unique classes and sort them so that 'non palm' is first
classes = np.sort(np.unique(Y))
classes = ['non palm'] + [c for c in classes if c != 'non palm']

# use np.unique to count occurrences of each class
unique, counts = np.unique(Y, return_counts=True)

class_counts = dict(zip(unique, counts))
print(class_counts)  # {'non-palm': 2, 'palm': 4}

# Define the preprocessing with augmentation
preprocess_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Define the preprocessing without augmentation
preprocess_test = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx].astype('uint8'), 'RGB')
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx], dtype=torch.long)  # Convert labels to LongTensor
        return image, label

# Initialize the model
def initialize_model():
    resnet = models.resnet18(pretrained=True)

    # Freeze the resnet layers
    for param in resnet.parameters():
        param.requires_grad = False

    # Replace the last fully-connected layer
    # Parameters of newly constructed modules have requires_grad=True by default
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Sequential(
                   # nn.Dropout(p=0.2),
                   nn.Linear(num_ftrs, nnodes),
                   nn.ReLU(),
                   nn.BatchNorm1d(nnodes),
                   nn.Linear(nnodes, 2),
                   nn.LogSoftmax(dim=1)
    )

    return resnet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running on device: ", device)

# Define a loss function and an optimizer
criterion = nn.NLLLoss()

# Encode labels into integers
le = LabelEncoder()
le.fit(classes)
Y_int = le.transform(Y).astype(np.int64)  # Convert to int64

# Generate indices for the train-test split
indices = np.arange(len(X))
indices_train, indices_test, Y_int_train, Y_int_test = train_test_split(indices, Y_int, test_size=0.2, random_state=42, stratify=Y)

# Create train and test datasets using the split indices
X_train, X_test, X_filenames_train, X_filenames_test = X[indices_train], X[indices_test], np.array(X_filenames)[indices_train], np.array(X_filenames)[indices_test]

# Define a batch size
batch_size = 64

# Define number of epochs
num_epochs = 500

# Initialize K-Fold
skf = StratifiedKFold(n_splits=5, shuffle=True)

# Prepare the lists for saving history
train_loss_list = []
val_loss_list = []
train_acc_list = []
val_acc_list = []

# Initialize the variable to keep track of the best epoch
best_epoch = -1

# Prepare to save the best model
best_model = None
best_val_loss = float('inf')

# Start K-Fold training
for fold, (train_ids, val_ids) in enumerate(skf.split(X_train, Y_int_train)):
    print(f'FOLD {fold+1}')
    print('--------------------------------')

    # Define the data subsets
    train_subs = np.array(X_train)[train_ids], np.array(Y_int_train)[train_ids]
    val_subs = np.array(X_train)[val_ids], np.array(Y_int_train)[val_ids]

    # Initialize custom datasets
    train_data = CustomImageDataset(train_subs[0], train_subs[1], transform=preprocess_train)
    val_data = CustomImageDataset(val_subs[0], val_subs[1], transform=preprocess_test)

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False)

    # Initialize the model for this fold
    resnet = initialize_model()
    resnet = resnet.to(device)
    optimizer = torch.optim.Adam(resnet.parameters(), lr=0.001)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=5, min_lr=0.00001)

    fold_best_val_loss = float('inf')
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        running_loss = 0.0
        running_corrects = 0
        resnet.train()
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward
            outputs = resnet(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Backward and optimize
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc.cpu().numpy())  # Transfer to CPU before conversion

        # Validate the model
        resnet.eval()
        running_loss = 0.0
        running_corrects = 0
        probabilities = []  # Initialize a list to store probabilities
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward
            outputs = resnet(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # Calculate probabilities
            probs = nn.functional.softmax(outputs, dim=1)  # Apply softmax to get probabilities
            probabilities.extend(probs.tolist())  # Add current batch probabilities to the list

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(val_loader.dataset)
        epoch_acc = running_corrects.double() / len(val_loader.dataset)

        print(f'Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        val_loss.append(epoch_loss)
        val_acc.append(epoch_acc.cpu().numpy())  # Transfer to CPU before conversion

        # Adjust learning rate if validation loss doesn't decrease for a long time
        scheduler.step(epoch_loss)

        # If the model improves, save a checkpoint
        if epoch_loss < fold_best_val_loss:
            fold_best_val_loss = epoch_loss
            if fold_best_val_loss < best_val_loss:
                best_val_loss = fold_best_val_loss
                best_model = resnet.state_dict()
                best_epoch = epoch
                print(f"Best model achieves at epoch {epoch+1} of fold {fold+1}")

    # Save the loss and accuracy history
    train_loss_list.append(train_loss)
    val_loss_list.append(val_loss)
    train_acc_list.append(train_acc)
    val_acc_list.append(val_acc)

# Save the best model
torch.save(best_model, f'{save_path}resnet18_model.pth')

# Plot the loss and accuracy
for i in range(5):
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss_list[i], label='Train Loss')
    plt.plot(val_loss_list[i], label='Val Loss')
    plt.legend()
    plt.title(f'Loss History of Fold {i+1}')

    plt.subplot(1, 2, 2)
    plt.plot(train_acc_list[i], label='Train Accuracy')
    plt.plot(val_acc_list[i], label='Val Accuracy')
    plt.legend()
    plt.title(f'Accuracy History of Fold {i+1}')
    plt.savefig(f"{save_path}Accuracy_Loss_Fold{i+1}.png")
    plt.show()

# Testing the model on the testing data set
test_data = CustomImageDataset(X_test, Y_int_test, transform=preprocess_test)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# Load the best model
resnet.load_state_dict(best_model)
resnet.eval()

# Create a DataLoader for the entire training set
misclassified_train_filenames = []
train_data_full = CustomImageDataset(X_train, Y_int_train, transform=preprocess_test)
train_loader_full = DataLoader(train_data_full, batch_size=batch_size, shuffle=False)


for inputs, labels in tqdm(train_loader_full):
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Forward
    outputs = resnet(inputs)
    _, preds = torch.max(outputs, 1)

    # If a prediction is incorrect, save the filename
    incorrect_indices = (preds != labels).nonzero(as_tuple=True)[0]  # Get the indices of incorrect predictions
    for idx in incorrect_indices:
        misclassified_train_filenames.append(X_filenames_train[idx])  # Assuming idx is the correct index in X_filenames_train

# Now, misclassified_train_filenames contains the filenames of the images that were misclassified in the training set
print("Misclassified training images:", misclassified_train_filenames)

# Initialize the variables
test_loss = 0.0
test_corrects = 0
Y_test_pred = []
Y_test_scores = []
Y_test_true = []
misclassified_test_filenames = []

for inputs, labels in tqdm(test_loader):
    inputs = inputs.to(device)
    labels = labels.to(device)

    # Forward
    outputs = resnet(inputs)
    _, preds = torch.max(outputs, 1)
    loss = criterion(outputs, labels)

    test_loss += loss.item() * inputs.size(0)
    test_corrects += torch.sum(preds == labels.data)

    # Save the true and predicted labels for computing metrics later
    Y_test_true.extend(labels.data.cpu().numpy())
    Y_test_pred.extend(preds.cpu().numpy())
    
    # If the prediction is incorrect, save the filename
    incorrect_indices = (preds != labels).nonzero(as_tuple=True)[0]  # Get the indices of incorrect predictions
    for idx in incorrect_indices:
        actual_batch_index = i * batch_size + idx  # Calculate the actual index in the dataset
        misclassified_test_filenames.append(X_filenames_test[actual_batch_index])

    # Calculate probabilities and scores
    probs = nn.functional.softmax(outputs, dim=1)  # Apply softmax to get probabilities
    Y_test_scores.extend(probs[:, 1].detach().cpu().numpy())  # Assuming binary classification

# Now, misclassified_test_filenames contains the filenames of the images that were misclassified in the testing set
print("Misclassified test images:", misclassified_test_filenames)

# Compute overall loss and accuracy for the testing set
test_loss = test_loss / len(test_loader.dataset)
test_acc = test_corrects.double() / len(test_loader.dataset)

print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')

# Compute ROC AUC score
roc_auc = roc_auc_score(Y_test_true, Y_test_scores)

# Compute confusion matrix
cm = confusion_matrix(Y_test_true, Y_test_pred)

# Compute average accuracy (mean of class-wise accuracy)
class_acc = cm.diagonal() / cm.sum(axis=1)
average_accuracy = np.mean(class_acc)

# Compute Cohen's kappa
kappa = cohen_kappa_score(Y_test_true, Y_test_pred)

# Compute precision and recall
cr = classification_report(Y_test_true, Y_test_pred, output_dict=True)
precision = cr['weighted avg']['precision']
recall = cr['weighted avg']['recall']

print(f'Test ROC AUC: {roc_auc}')
print(f'Test Average Accuracy: {average_accuracy}')
print(f'Test Kappa: {kappa}')
print(f'Test Precision: {precision}')
print(f'Test Recall: {recall}')

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Ensure Y_test and Y_test_scores are numpy arrays
Y_test = np.array(Y_int_test)
Y_test_scores = np.array(Y_test_scores)

# Compute ROC curve and ROC area
fpr, tpr, thresholds = roc_curve(Y_test, Y_test_scores)
roc_auc = auc(fpr, tpr)

# Calculate the G-mean for each threshold
gmeans = np.sqrt(tpr * (1-fpr))

# Locate the index of the largest G-mean
ix = np.argmax(gmeans)

# Locate the optimal threshold
optimal_threshold = thresholds[ix]

print(f'Optimal Threshold based on ROC curve: {optimal_threshold}')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

lw=2

# Plot ROC curve
ax1.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
ax1.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')

ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.0])
ax1.set_xlabel('False Positive Rate')
ax1.set_ylabel('True Positive Rate')
ax1.set_title('Receiver Operating Characteristic')
ax1.legend(loc="lower right")

# Scores corresponding to class 0
Y_scores_0 = Y_test_scores[Y_int_test==0]

# Scores corresponding to class 1
Y_scores_1 = Y_test_scores[Y_int_test==1]

ax2.hist(Y_scores_0, bins=50, alpha=0.5, label='Class 0 (non palm)')
ax2.hist(Y_scores_1, bins=50, alpha=0.5, label='Class 1 (palm)')
ax2.set_xlabel('Score')
ax2.set_ylabel('Frequency')
ax2.set_title('Distribution of Scores for Each Class')
ax2.legend(loc='upper right')

plt.tight_layout()
plt.savefig(f"{path}figures_{nnodes}_updated/ROC_curves.png")
plt.show()

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
model.load_state_dict(torch.load(f'{save_path}resnet18_model.pth'))
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
    np.save(f"{save_path}small_image_{i}_heatmap_prob.npy", average_predictions_resized)

    # Convert to float32 before normalize
    average_predictions_resized = average_predictions_resized.astype(np.float32)

    # Convert the probability heatmap to a mask
    mask_prob = (average_predictions_resized * 255).astype(np.uint8)

    # Convert PIL Image to numpy array
    small_image = np.array(small_image)

    # Highlight the palms in the image using the probability mask
    highlighted_image_prob = cv2.addWeighted(small_image, 0.5, cv2.cvtColor(mask_prob, cv2.COLOR_GRAY2BGR), 0.5, 0)

    # Save the probability highlighted image
    cv2.imwrite(f"{save_path}small_image_{i}_highlighted_prob.jpg", highlighted_image_prob)

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
        small_image_highlighted_prob = Image.open(f"{save_path}small_image_{i*num_images_height + j}_highlighted_prob.jpg")
        large_image_highlighted_prob.paste(small_image_highlighted_prob, (i*small_image_size[0], j*small_image_size[1]))
        # Place the small heatmap onto the large heatmap canvas
        small_heatmap_prob = np.load(f"{save_path}small_image_{i*num_images_height + j}_heatmap_prob.npy")
        large_heatmap_prob[j*small_image_size[1]:(j+1)*small_image_size[1], i*small_image_size[0]:(i+1)*small_image_size[0]] = small_heatmap_prob

# Crop the combined image to the original image size
large_image_highlighted_prob = large_image_highlighted_prob.crop((0, 0, large_image.width, large_image.height))

# Crop the combined heatmap to the original image size
large_heatmap_prob = large_heatmap_prob[0:large_image.height, 0:large_image.width]

# Save the resulting large image
large_image_highlighted_prob.save(f"{save_path}FCAT2APPK_highlighted_prob.jpg")

# Save the resulting large heatmap
np.save(f"{save_path}FCAT2APPK_heatmap_prob.npy", large_heatmap_prob)