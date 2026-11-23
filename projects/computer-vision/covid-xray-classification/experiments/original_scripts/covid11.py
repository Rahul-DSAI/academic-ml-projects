import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Define constants
IMAGE_SIZE = 256
BATCH_SIZE = 8
CHANNELS = 3
dropout_rate = 0.6

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

# Model
model = keras.Sequential([
    layers.InputLayer(input_shape=(IMAGE_SIZE, IMAGE_SIZE, CHANNELS)),
    layers.experimental.preprocessing.Rescaling(1./255),
    layers.Conv2D(32, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(dropout_rate),
    layers.Dense(512, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
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
test_loss, test_acc = model.evaluate(test_ds)
print('\nTest accuracy:', test_acc)
