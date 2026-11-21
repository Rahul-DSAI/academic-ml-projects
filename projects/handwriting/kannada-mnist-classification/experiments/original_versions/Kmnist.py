from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
from keras.callbacks import LearningRateScheduler
import numpy as np
import matplotlib.pyplot as plt

batch_size = 128
num_classes = 10
epochs = 50  # Increased epochs

# Learning rate scheduler
def lr_scheduler(epoch):
    return 0.001 * np.exp(-epoch / 10.)  # Adjusted learning rate

# Load the data
def load(f):
    return np.load(f)['arr_0']

x_train = load("D:\\Downloads\\archive (3)\\Kannada_MNIST_datataset_paper\\Kannada_MNIST_npz\\Kannada_MNIST\\X_kannada_MNIST_train.npz")
x_test = load("D:\\Downloads\\archive (3)\\Kannada_MNIST_datataset_paper\\Kannada_MNIST_npz\\Kannada_MNIST\\X_kannada_MNIST_test.npz")
y_train = load("D:\\Downloads\\archive (3)\\Kannada_MNIST_datataset_paper\\Kannada_MNIST_npz\\Kannada_MNIST\\y_kannada_MNIST_train.npz")
y_test = load("D:\\Downloads\\archive (3)\\Kannada_MNIST_datataset_paper\\Kannada_MNIST_npz\\Kannada_MNIST\\y_kannada_MNIST_test.npz")

input_shape = x_train.shape[1:] + (1,)

x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

print('{} train samples, {} test samples'.format(len(x_train), len(x_test)))

y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

# Learning rate scheduler callback
lr_scheduler_callback = LearningRateScheduler(lr_scheduler)

# Fit the model with learning rate scheduling, batch normalization, and dropout
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test),
          callbacks=[lr_scheduler_callback])

train_score = model.evaluate(x_train, y_train, verbose=0)
test_score = model.evaluate(x_test, y_test, verbose=0)
print('Train loss:', train_score[0])
print('Train accuracy:', train_score[1])
print('Test loss:', test_score[0])
print('Test accuracy:', test_score[1])

# Predicting classes for test data
predictions = model.predict(x_test)

# Converting predictions to class labels
predicted_classes = np.argmax(predictions, axis=1)

# Displaying the predicted classes
print("Predicted classes for test data:")
print(predicted_classes)
