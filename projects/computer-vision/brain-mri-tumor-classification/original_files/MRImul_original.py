import tensorflow as tf
from tensorflow.keras import layers, Sequential

# Define constants
batch_size = 16
num_classes = 4
epochs = 20

# Load dataset without shuffling
trainset = tf.keras.preprocessing.image_dataset_from_directory(
    "C:\\Users\\rahul\\Downloads\\archive (1)\\Training",
    shuffle=False,  # Disable shuffling
    batch_size=batch_size,
    image_size=(64, 64),
    seed=42
)
testset = tf.keras.preprocessing.image_dataset_from_directory(
    "C:\\Users\\rahul\\Downloads\\archive (1)\\Testing",
    shuffle=False,  # Disable shuffling
    batch_size=batch_size,
    image_size=(64, 64),
    seed=42
)

# Rescale pixel values
trainset = trainset.map(lambda x, y: (x / 255.0, y))
testset = testset.map(lambda x, y: (x / 255.0, y))

# Split dataset
trainset = trainset.shuffle(buffer_size=len(trainset), seed=42)  # Shuffle before splitting
train_length = len(trainset)
val_length = int(0.1 * train_length)
train_length = train_length - val_length
train_ds = trainset.take(train_length)
val_ds = trainset.skip(train_length).take(val_length)
test_ds = testset

# Model
model = Sequential([
    layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)),
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
