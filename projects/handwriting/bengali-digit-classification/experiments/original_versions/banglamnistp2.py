import tensorflow as tf
from tensorflow.keras import layers, Sequential
import matplotlib.pyplot as plt

# Define constants
batch_size = 64  # Reduced batch size
num_classes = 10
epochs = 30

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "data/raw/bengali_digits",
    shuffle=True,
    batch_size=batch_size,
    image_size=(28, 28),
    seed=42,
    labels='inferred',  # This will use directory names as class labels
    label_mode='int',   # Use integer labels
    class_names=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']  # Specify class names
)

# Rescale pixel values
dataset = dataset.map(lambda x, y: (x / 255.0, y))

# Calculate dataset lengths
total_length = len(dataset)
train_length = int(0.7 * total_length)
val_length = int(0.1 * total_length)
test_length = total_length - train_length - val_length

# Split dataset
train_ds = dataset.take(train_length)
val_ds = dataset.skip(train_length).take(val_length)
test_ds = dataset.skip(train_length + val_length).take(test_length)

# Model
model = Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 3)),
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

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
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
train_score = model.evaluate(train_ds, verbose=0)
test_score = model.evaluate(test_ds, verbose=0)
print('Train loss:', train_score[0])
print('Train accuracy:', train_score[1])
print('Test loss:', test_score[0])
print('Test accuracy:', test_score[1])
