import os
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img

# Define the parameters for data augmentation
datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

# Directory containing your images
directory = "C:\\Users\\rahul\\Downloads\\COVID-19\\COVID"
# List all files in the directory
files = os.listdir(directory)

# Loop through each file
for file in files:
    # Load the image
    img = load_img(os.path.join(directory, file))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)

    # Generate augmented images and save them
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                              save_to_dir=directory, save_prefix='COVID' + file[:-4], save_format='png'):
        i += 1
        if i > 1200:  # Generate 20 augmented images per original image
            break