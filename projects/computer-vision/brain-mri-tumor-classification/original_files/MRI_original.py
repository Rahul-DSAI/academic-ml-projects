import tensorflow as tf
from tensorflow.keras import layers, Sequential

# Define constants
batch_size = 16
num_classes = 4
epochs = 60

# Load dataset without shuffling
trainset = tf.keras.preprocessing.image_dataset_from_directory(
    "C:\\Users\\rahul\\Downloads\\archive (1)\\Training",
    shuffle=True,
    batch_size=batch_size,
    image_size=(128, 128),
    seed=42
)

testset = tf.keras.preprocessing.image_dataset_from_directory(
    "C:\\Users\\rahul\\Downloads\\archive (1)\\Testing",
    shuffle=False,
    batch_size=batch_size,
    image_size=(128, 128),
    seed=42
)

# Rescale pixel values
trainset = trainset.map(lambda x, y: (x / 255.0, y))
testset = testset.map(lambda x, y: (x / 255.0, y))

# Calculate dataset lengths
train_length = len(trainset)
val_length = int(0.1 * train_length)
train_length = train_length - val_length

# Split dataset
trainset = trainset.shuffle(buffer_size=train_length + val_length, seed=42)
train_ds = trainset.take(train_length)
val_ds = trainset.skip(train_length).take(val_length)

# Model
model = Sequential([
    layers.Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),

    layers.Flatten(),

    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),

    layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Display model summary
model.summary()

# Train the model
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(testset)
print('Test loss:', test_loss)
print('Test accuracy:', test_accuracy)

train_loss, train_accuracy = model.evaluate(train_ds)
print('Train loss:', train_loss)
print('Train accuracy:', train_accuracy)
