import os
import random
from PIL import Image
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

# Directory containing your images
directory = "C:\\Users\\rahul\\Downloads\\COVID-19\\Viral Pneumonia"
# List all files in the directory
files = os.listdir(directory)

# Select a random image file
random_file = random.choice(files)

# Load the randomly selected image
img = Image.open(os.path.join(directory, random_file))
x = np.array(img)
x = np.expand_dims(x, axis=0)  # Add batch dimension
x = np.expand_dims(x, axis=-1)  # Add channel dimension for grayscale image

# Define the parameters for data augmentation
datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# Generate augmented images and save them
i = 0
for batch in datagen.flow(x, batch_size=1,
                          save_to_dir=directory, save_prefix='Viral Pneumonia', save_format='png'):
    i += 1
    if i >= 1500:  # Generate 20 augmented images
        break
