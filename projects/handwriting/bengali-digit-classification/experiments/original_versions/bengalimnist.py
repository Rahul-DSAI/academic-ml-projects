import tensorflow as tf
from tensorflow.keras import layers, Sequential

# Define constants
batch_size = 64  # Reduced batch size
num_classes = 10
epochs = 30

dataset = tf.keras.preprocessing.image_dataset_from_directory(
    "data/raw/pust_bengali_digits",
    shuffle=True,
    batch_size=batch_size,
    image_size=(28, 28),
    seed=42
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
    layers.Dropout(0.2),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.Conv2D(256, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.4),
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

# Evaluate the model
train_score = model.evaluate(train_ds, verbose=0)
test_score = model.evaluate(test_ds, verbose=0)
print('Train loss:', train_score[0])
print('Train accuracy:', train_score[1])
print('Test loss:', test_score[0])
print('Test accuracy:', test_score[1])
