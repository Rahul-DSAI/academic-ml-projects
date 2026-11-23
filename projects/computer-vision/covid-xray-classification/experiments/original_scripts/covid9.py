import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define constants
IMAGE_SIZE = 299
BATCH_SIZE = 16
CHANNELS = 3

# Load dataset
dataset = keras.preprocessing.image_dataset_from_directory(
    "C:\\Users\\rahul\\Downloads\\COVID-19",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE
)

# Calculate dataset lengths
total_length = len(dataset)
train_length = int(0.8 * total_length)
val_length = int(0.1 * total_length)
test_length = total_length - train_length - val_length

# Split dataset
train_ds = dataset.take(train_length)
val_ds = dataset.skip(train_length).take(val_length)
test_ds = dataset.skip(train_length + val_length).take(test_length)

# Data augmentation
data_augmentation = keras.Sequential([
    layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    layers.experimental.preprocessing.RandomRotation(0.2),
    layers.experimental.preprocessing.RandomZoom(0.2),
])

# Model
model = keras.Sequential([
    layers.InputLayer(input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
    data_augmentation,
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.35),
    layers.Dense(512, activation='relu'),
    layers.Dense(4, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Learning rate reduction callback
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.6,
    patience=2,
    min_lr=1e-6
)

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    callbacks=[reduce_lr]  # Add the reduce_lr callback here
)

# Evaluate the model
test_loss, test_acc = model.evaluate(test_ds)
print('\nTest accuracy:', test_acc)
