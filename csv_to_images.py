import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from IPython.display import display

# Assuming your CSV file is named 'english_dataset.csv'
csv_file_path = 'english_alphabets_data/A_Z Handwritten Data.csv'
output_folder = 'english_alphabets_data/'

os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.join(output_folder, 'train'), exist_ok=True)
os.makedirs(os.path.join(output_folder, 'test'), exist_ok=True)

# Load Data with custom column names
names = ['class']
for id in range(1, 785):
    names.append(id)

df = pd.read_csv(csv_file_path, header=None, names=names)
display(df.head(5))

# Class mapping
class_mapping = {}
alphabets = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
for i in range(len(alphabets)):
    class_mapping[i] = alphabets[i]

# Map class labels to alphabets
df['class'] = df['class'].map(class_mapping)

# Extract features (pixel values) and labels
X = df.drop('class', axis=1)
y = df['class']

# Reshape the data to 28x28 pixels
X_reshaped = X.values.reshape(-1, 28, 28)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Save images to folders with subfolders for each class
def save_images(images, labels, folder_path, subfolder):
    for i, (image, label) in enumerate(zip(images, labels)):
        class_folder_path = os.path.join(folder_path, subfolder, label)
        os.makedirs(class_folder_path, exist_ok=True)
        image_path = os.path.join(class_folder_path, f"{label}_{i}.png")
        plt.imsave(image_path, image, cmap='gray')
        print(f"done {i}")

# Save training images with subfolders for each class
save_images(X_train, y_train, os.path.join(output_folder, 'train'), 'train')

# Save testing images with subfolders for each class
save_images(X_test, y_test, os.path.join(output_folder, 'test'), 'test')
