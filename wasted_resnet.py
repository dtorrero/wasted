# Import required libraries
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
import kagglehub
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set up matplotlib for better visualization
plt.style.use('default')
plt.rcParams['figure.figsize'] = [15, 10]

# Download the dataset using kagglehub
path = kagglehub.dataset_download("asdasdasasdas/garbage-classification")
print("Path to dataset files:", path)

# Define the base path to the dataset
base_path = Path(path) / 'Garbage classification'

# Create lists to store the data
image_paths = []
categories = []

# Walk through the directory and collect all image paths and their categories
for category in os.listdir(base_path):
    category_path = base_path / category
    if os.path.isdir(category_path):
        for image_file in os.listdir(category_path):
            if image_file.endswith('.jpg'):
                image_paths.append(str(category_path / image_file))
                categories.append(category)

# Create the DataFrame
df = pd.DataFrame({
    'image_path': image_paths,
    'category': categories
})

# Convert categories to numerical labels
category_to_label = {category: idx for idx, category in enumerate(df['category'].unique())}
df['label'] = df['category'].map(category_to_label)

# Split the data into train and validation sets
from sklearn.model_selection import train_test_split
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['category'])

# Data preprocessing
def preprocess_image(image_path, label):
    # Read the image file
    img = tf.io.read_file(image_path)
    # Decode the image
    img = tf.image.decode_jpeg(img, channels=3)
    # Resize the image
    img = tf.image.resize(img, [224, 224])
    # Normalize the image
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((train_df['image_path'].values, train_df['label'].values))
train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

val_dataset = tf.data.Dataset.from_tensor_slices((val_df['image_path'].values, val_df['label'].values))
val_dataset = val_dataset.map(preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
val_dataset = val_dataset.batch(32).prefetch(tf.data.AUTOTUNE)

# Create the model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(category_to_label), activation='softmax')
])

# Compile the model
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stop = callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2,
    min_lr=1e-6
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=30,
    callbacks=[early_stop, reduce_lr]
)

# Plot training history
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Save the model
model.save('waste_classification_model.h5')

# Display basic information about the dataset
print("Dataset Information:")
print(f"Total number of images: {len(df)}")
print("\nNumber of images per category:")
print(df['category'].value_counts())

# Display the first few rows of the DataFrame
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Create a figure with subplots for each category
plt.figure(figsize=(15, 10))

# Get one example from each category
for idx, category in enumerate(df['category'].unique()):
    # Get the first image path for this category
    image_path = df[df['category'] == category]['image_path'].iloc[0]
    
    # Create subplot
    plt.subplot(2, 3, idx + 1)
    
    # Read and display the image
    img = Image.open(image_path)
    plt.imshow(img)
    plt.title(category, fontsize=12, pad=10)
    plt.axis('off')

plt.tight_layout()
plt.show()

# Optional: Display some statistics about image sizes
print("\nImage Statistics:")
image_sizes = []
for path in df['image_path']:
    img = Image.open(path)
    image_sizes.append(img.size)

sizes_df = pd.DataFrame(image_sizes, columns=['width', 'height'])
print("\nImage dimensions statistics:")
print(sizes_df.describe()) 