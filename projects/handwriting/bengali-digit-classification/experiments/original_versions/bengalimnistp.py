import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt

# Define constants
batch_size = 64
num_classes = 10
epochs = 100
image_size = (28, 28)

# Function to load TIFF images and assign labels
def load_images_and_labels(directory):
    images = []
    labels = []
    class_names = sorted(os.listdir(directory))
    for class_name in class_names:
        class_directory = os.path.join(directory, class_name)
        for filename in os.listdir(class_directory):
            if filename.endswith('.tif'):
                image_path = os.path.join(class_directory, filename)
                image = Image.open(image_path)
                image = image.resize(image_size)  # Resize image to (28, 28)
                image = np.array(image) / 255.0  # Normalize pixel values
                images.append(image)
                labels.append(class_names.index(class_name))
    return np.array(images), np.array(labels)

# Load images with labels
images, labels = load_images_and_labels("data/raw/tiff/train")

# Shuffle data
indices = np.random.permutation(len(images))
images, labels = images[indices], labels[indices]

# Split data into train, validation, and test sets
train_size = int(0.9 * len(images))
val_size = int(0.1 * len(images))
test_size = load_images_and_labels("data/raw/tiff/test")

train_images, train_labels = images[:train_size], labels[:train_size]
val_images, val_labels = images[train_size:train_size+val_size], labels[train_size:train_size+val_size]
test_images, test_labels = images[train_size+val_size:], labels[train_size+val_size:]

# Data augmentation using ImageDataGenerator
datagen = ImageDataGenerator(
    rotation_range=20,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # shear_range=0.1,
    zoom_range=0.1,
    # horizontal_flip=True,
    # vertical_flip=True,
    # fill_mode='nearest'
)

# Model
model = Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 1)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
# Print model summary
model.summary()

# Fit the model with data augmentation
history = model.fit(
    datagen.flow(train_images.reshape(-1, 28, 28, 1), train_labels, batch_size=batch_size),
    steps_per_epoch=len(train_images) // batch_size,
    epochs=epochs,
    validation_data=(val_images.reshape(-1, 28, 28, 1), val_labels)
    # verbose=1
)

# Plot training and validation graphs
# Get training history
train_loss = history.history['loss']
train_accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']

# Plot training and validation loss
plt.figure(figsize=(10, 5))
plt.plot(train_loss, label='Training Loss', color='blue')
plt.plot(val_loss, label='Validation Loss', color='red')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot training and validation accuracy
plt.figure(figsize=(10, 5))
plt.plot(train_accuracy, label='Training Accuracy', color='blue')
plt.plot(val_accuracy, label='Validation Accuracy', color='red')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Evaluate the model
test_score = model.evaluate(test_images.reshape(-1, 28, 28, 1), test_labels, verbose=0)
print('Test loss:', test_score[0])
print('Test accuracy:', test_score[1])
