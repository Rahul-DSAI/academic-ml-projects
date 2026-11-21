from __future__ import print_function
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization
import numpy as np

batch_size = 128
num_classes = 10
epochs = 20

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
model.add(Dropout(0.15))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.15))
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.3))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

train_score = model.evaluate(x_train, y_train, verbose=0)
test_score = model.evaluate(x_test, y_test, verbose=0)
print('Train loss:', train_score[0])
print('Train accuracy:', train_score[1])
print('Test loss:', test_score[0])
print('Test accuracy:', test_score[1])

y_train_pred = model.predict(x_train)
y_test_pred = model.predict(x_test)

y_train_pred_classes = np.argmax(y_train_pred, axis=1)
y_test_pred_classes = np.argmax(y_test_pred, axis=1)

y_train_true_classes = np.argmax(y_train, axis=1)
y_test_true_classes = np.argmax(y_test, axis=1)

train_class_accuracy = {}
test_class_accuracy = {}

for class_label in range(num_classes):
    train_indices = np.where(y_train_true_classes == class_label)[0]
    test_indices = np.where(y_test_true_classes == class_label)[0]
    
    train_correct = np.sum(y_train_pred_classes[train_indices] == y_train_true_classes[train_indices])
    test_correct = np.sum(y_test_pred_classes[test_indices] == y_test_true_classes[test_indices])
    
    train_accuracy = train_correct / len(train_indices) if len(train_indices) > 0 else 0.0
    test_accuracy = test_correct / len(test_indices) if len(test_indices) > 0 else 0.0
    
    train_class_accuracy[class_label] = train_accuracy
    test_class_accuracy[class_label] = test_accuracy

print("\nClass-wise training accuracies:")
for class_label, accuracy in train_class_accuracy.items():
    print("Class {}: {:.2f}%".format(class_label, accuracy * 100))

print("\nClass-wise testing accuracies:")
for class_label, accuracy in test_class_accuracy.items():
    print("Class {}: {:.2f}%".format(class_label, accuracy * 100))
