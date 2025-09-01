import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
import numpy as np

# Set up data directories
train_dir = 'data/train'
validation_dir = 'data/validation'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load data from the directories
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    labels='inferred',
    label_mode='categorical',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

# 1. Load the pre-trained VGG16 base model
base_model = VGG16(
    weights='imagenet',
    include_top=False,
    input_shape=(224, 224, 3)
)

# 2. Freeze the base model's weights
base_model.trainable = False

# 3. Create a new model with the VGG16 base and a custom classification head
num_classes = len(train_dataset.class_names)
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# 4. Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 5. Train the model
epochs = 10  # You can adjust the number of epochs
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=validation_dataset
)

# Optional: You can save the trained model for future use
model.save('vgg_transfer_model.h5')