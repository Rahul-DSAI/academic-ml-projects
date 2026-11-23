import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define constants
IMAGE_SIZE = 256
BATCH_SIZE = 32
CHANNELS = 3

# Load dataset
train_ds = keras.preprocessing.image_dataset_from_directory(
    "C:\\Users\\rahul\\Downloads\\COVID-19",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="training",
    seed=42  # Add seed argument to ensure consistent shuffling
)

val_ds = keras.preprocessing.image_dataset_from_directory(
    "C:\\Users\\rahul\\Downloads\\COVID-19",
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset="validation",
    seed=42  # Add seed argument to ensure consistent shuffling
)

# Model
model = keras.Sequential([
    layers.InputLayer(input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
    layers.Rescaling(1./255),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),  # Dropout layer added after each pooling layer

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),

    layers.GlobalAveragePooling2D(),

    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(4, activation='softmax')
])

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50
)

# Evaluate the model
test_loss, test_acc = model.evaluate(val_ds)
print('\nValidation accuracy:', test_acc)

train_loss, train_acc = model.evaluate(train_ds)
print('\nTraining accuracy:', train_acc)
