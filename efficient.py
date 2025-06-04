import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, applications, mixed_precision
from tensorflow.keras.applications import EfficientNetB0  # More efficient than ResNet
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, 
                                      ReduceLROnPlateau, TensorBoard, CSVLogger)
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import RandomContrast, RandomZoom, RandomRotation

# Enable mixed precision for better GPU utilization
mixed_precision.set_global_policy('mixed_float16')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

# Configuration
IMG_SIZE = (300, 300)  # Increased from 224 for better feature extraction
BATCH_SIZE = 32  # Reduced batch size for stability
EPOCHS = 50
PATIENCE = 10
INIT_LR = 0.001
MIN_LR = 1e-6

# Data loading and preparation
base_path = Path('data/garbage classification/Garbage classification')

# Create DataFrame
image_paths = []
categories = []
for category in os.listdir(base_path):
    category_path = base_path / category
    if os.path.isdir(category_path):
        for img_file in os.listdir(category_path):
            if img_file.endswith(('.jpg', '.jpeg', '.png')):
                image_paths.append(str(category_path / img_file))
                categories.append(category)

df = pd.DataFrame({'image_path': image_paths, 'category': categories})
label_map = {label: idx for idx, label in enumerate(df['category'].unique())}
df['label'] = df['category'].map(label_map)

# Handle class imbalance
class_weights = compute_class_weight('balanced', classes=np.unique(df['label']), y=df['label'])
class_weights = dict(enumerate(class_weights))

# Data splitting
train_df, temp_df = train_test_split(df, test_size=0.2, stratify=df['category'], random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['category'], random_state=42)

# Enhanced data augmentation
def custom_augmentation(image):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.8, 1.2)
    image = tf.image.random_saturation(image, 0.8, 1.2)
    image = tf.image.random_hue(image, 0.1)
    return image

train_datagen = ImageDataGenerator(
    preprocessing_function=applications.efficientnet.preprocess_input,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='reflect',
    brightness_range=[0.7, 1.3],
    validation_split=0.0
)

val_datagen = ImageDataGenerator(
    preprocessing_function=applications.efficientnet.preprocess_input
)

# Data generators with balanced batches
def create_balanced_generator(df, datagen, batch_size):
    return datagen.flow_from_dataframe(
        dataframe=df,
        x_col='image_path',
        y_col='label',
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode='raw',
        shuffle=True,
        seed=42
    )

train_gen = create_balanced_generator(train_df, train_datagen, BATCH_SIZE)
val_gen = create_balanced_generator(val_df, val_datagen, BATCH_SIZE)
test_gen = create_balanced_generator(test_df, val_datagen, BATCH_SIZE)

# Model architecture with EfficientNet
def build_model():
    base_model = EfficientNetB0(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        pooling='avg'
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    
    # Enhanced head
    x = layers.Dense(512, activation='swish', kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.5)(x)
    
    x = layers.Dense(256, activation='swish', kernel_regularizer=l2(0.01))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    outputs = layers.Dense(len(label_map), activation='softmax', dtype='float32')(x)
    
    model = models.Model(inputs, outputs)
    
    optimizer = optimizers.AdamW(
        learning_rate=INIT_LR,
        weight_decay=0.0001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-07
    )
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy',
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2, name='top2_accuracy')
        ]
    )
    
    return model

model = build_model()
model.summary()

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=MIN_LR),
    ModelCheckpoint('best_model.h5', monitor='val_accuracy', save_best_only=True),
    TensorBoard(log_dir='./logs'),
    CSVLogger('training_log.csv')
]

# Training
history = model.fit(
    train_gen,
    steps_per_epoch=len(train_gen),
    validation_data=val_gen,
    validation_steps=len(val_gen),
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

# Fine-tuning
base_model = model.layers[1]
base_model.trainable = True

# Freeze first 150 layers
for layer in base_model.layers[:150]:
    layer.trainable = False

model.compile(
    optimizer=optimizers.Adam(learning_rate=INIT_LR/10),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy', tf.keras.metrics.SparseTopKCategoricalAccuracy(k=2)]
)

history_fine = model.fit(
    train_gen,
    steps_per_epoch=len(train_gen),
    validation_data=val_gen,
    validation_steps=len(val_gen),
    epochs=EPOCHS + 20,
    initial_epoch=history.epoch[-1],
    callbacks=callbacks,
    class_weight=class_weights
)

# Evaluation
def evaluate_model(model, test_gen):
    test_gen.reset()
    y_true = test_gen.labels
    y_pred = model.predict(test_gen)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred_classes, target_names=label_map.keys()))
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_true, y_pred_classes)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_map.keys(), yticklabels=label_map.keys())
    plt.title('Confusion Matrix')
    plt.show()

evaluate_model(model, test_gen)

# Save final model
model.save('garbage_classifier_enhanced.h5')