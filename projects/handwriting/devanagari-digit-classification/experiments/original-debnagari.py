import tensorflow as tf
from tensorflow.keras import layers, Sequential
import numpy as np

# Define constants
batch_size = 16
num_classes = 10
epochs = 40

# Load training dataset from training-images.npz
with np.load("C:\\Users\\rahul\\Downloads\\CMATERdb-master\\CMATERdb-master\\datasets\\devanagari-numerals\\training-images.npz") as train_data:
    keys = list(train_data.keys())  # Convert KeysView object to list
    x_train, y_train = train_data[keys[0]], train_data[keys[1]]  # Access data using keys

# Load testing dataset from testing-images.npz
with np.load("C:\\Users\\rahul\\Downloads\\CMATERdb-master\\CMATERdb-master\\datasets\\devanagari-numerals\\testing-images.npz") as test_data:
    keys = list(test_data.keys())  # Convert KeysView object to list
    x_test, y_test = test_data[keys[0]], test_data[keys[1]]  # Access data using keys

# Rescale pixel values
x_train, x_test = x_train / 255.0, x_test / 255.0

# Model
model = Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.BatchNormalization(),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    # layers.Dropout(0.1),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    # layers.Dropout(0.1),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.2),
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

# Data augmentation
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create an ImageDataGenerator instance for data augmentation
datagen = ImageDataGenerator(
    rotation_range=10,  # Randomly rotate images by up to 10 degrees
    width_shift_range=0.1,  # Randomly shift images horizontally by up to 10% of the width
    height_shift_range=0.1,  # Randomly shift images vertically by up to 10% of the height
    zoom_range=0.1,  # Randomly zoom images by up to 10%
    horizontal_flip=True,  # Randomly flip images horizontally
    vertical_flip=False,  # Do not flip images vertically
    fill_mode='nearest'  # Fill in newly created pixels using the nearest existing pixel
)

# Fit the ImageDataGenerator to your training data
datagen.fit(x_train)

# Train the model using augmented data
history = model.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    epochs=epochs,
)

# Evaluate the model
test_score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', test_score[0])
print('Test accuracy:', test_score[1])

# Calculate overall training accuracy
overall_train_accuracy = np.mean(history.history['accuracy'])
print('Overall training accuracy:', overall_train_accuracy)

