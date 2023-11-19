# Import ImageDataGenertor class from TensorFlow API
from keras.preprocessing.image import ImageDataGenerator

#Construct an instance of the ImageDataGenerator
datagen = ImageDataGenerator(
  rotation_range=45,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  fill_mode='nearest')

from skimage import io

#Load the image
x = io.imread('cat.jpg')
x.shape

#Reshape the image since datagen.flow accepts NumPy array of rank 4
x = x.reshape((1, ) + x.shape) #Array with shape (1, W, H, 3)
x.shape

i = 0
# Only one image
for batch in datagen.flow(x, batch_size=16, save_to_dir='augmented', save_prefix='aug', save_format='png'):
    i += 1
    if i > 20:
        break  # To limit the number of generated images

#Apply image augmentation
#a. vertical_flip
#b. brightness_range
#c. channel_shift_range

# Load the image
x = io.imread('cat.jpg')

# Reshape the image for augmentation
x = x.reshape((1, ) + x.shape)  # Array with shape (1, W, H, 3)

# Create an ImageDataGenerator with augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    vertical_flip=True,  # Apply vertical flip
    brightness_range=[0.5, 1.5],  # Adjust brightness within the specified range
    channel_shift_range=50  # Shift color channels
)

# Generate augmented images and save them to the 'augmented' folder
i = 0
# Only one image
for batch in datagen.flow(x, batch_size=16, save_to_dir='augmented', save_prefix='aug', save_format='png'):
    i += 1
    if i > 20:
        break  # To limit the number of generated images


