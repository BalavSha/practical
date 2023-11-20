# Batch Normalization
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten,Activation
from tensorflow.keras.layers import BatchNormalization
import matplotlib.pyplot as plt

tf.keras.datasets.mnist.load_data(path="mnist.npz")

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

plt.imshow(X_train[12], cmap = plt.get_cmap('gray'))

X_train = X_train / 255.0
X_test = X_test / 255.0

# without batch Norm
model = Sequential()
model.add(Flatten(input_shape=[28, 28]))
model.add(Dense(300, activation = 'relu'))
model.add(Dense(100, activation = 'relu')
model.add(Dense(10, activation = 'softmax'))

optimizer = tf.keras.optimizers.Adam()
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_split=0.2, verbose=2)

model.evaluate(X_train, y_train)
model.evaluate(X_test, y_test)

"We need to normalize the X_train and X_test dara to ensure that the data in X train and X test are of the same scale and 
prevent the algorithm from making assumptions about the distribution of data."

# With Batch Normalization
model = Sequential()
model.add(Flatten(input_shape=[28, 28]))
model.add(Dense(300, activation = 'relu'))
model.add(Dense(100, activation = 'relu'))
model.add(Dense(10, activation = 'softmax'))
model.add(BatchNormalization())
model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, validation_split=0.2, verbose=2)

model.evaluate(X_train, y_train)
model.evaluate(X_test, y_test)

# Batch Norm after hidden layer but before the activation function
model = Sequential()
model.add(Flatten(input_shape=[28, 28]))
model.add(Dense(300))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(100))
model.add(BatchNormalization())
model.add(Activation('relu'))

model.add(Dense(10))
model.add(BatchNormalization())
model.add(Activation('softmax'))

model.summary()

# Diabetics Prediction
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load the dataset
dataset = loadtxt('pima-indians-diabetes.csv', delimiter=',')

# Split features and labels
X = dataset[:, 0:8]
y = dataset[:, 8]

# Data preprocessing: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Print the shape of X and y
print("Shape of X: {}".format(X_scaled.shape))
print("Shape of y: {}".format(y.shape))

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Define the Keras model
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Print summary of the model
model.summary()

# Visualize the model
from tensorflow.keras.utils import plot_model
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

# Compile the Keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model on the training data
history = model.fit(X_train, y_train, epochs=150, batch_size=10, validation_data=(X_test, y_test))

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Evaluate the model on the testing set
_, accuracy = model.evaluate(X_test, y_test)
print('Test Accuracy: %.2f' % (accuracy * 100))

# Plot Confusion Matrix and Classification Report
y_pred = model.predict_classes(X_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

classification_rep = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_rep)

# Plot ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, model.predict_proba(X_test))
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# HeartDisease
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split

df = pd.read_csv("/content/heart.csv")
df.head()
df.shape
df.describe()

df.isnull().sum()
The **isnull().sum()** function to check for and count the number of null (missing) values in each column of a Pandas DataFrame. <br>
We use sum() function after isnull() **to aggregate the results of isnull() for each column.**

x_inputs = df.iloc[:, :-1]  # Select all columns except the last one as features
target = df.iloc[:, -1]

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(x_inputs, target, test_size=0.2, random_state=42)

target.unique()

# Create a Sequential model
model = Sequential()
# Add the first hidden layer with 8 neurons and specify the input shape (assuming you have input features)
model.add(Dense(units=8, activation='relu', input_shape=(13,)))
# Add the second hidden layer with 12 neurons
model.add(Dense(units=12, activation='relu'))
# Add the third hidden layer with 14 neurons
model.add(Dense(units=14, activation='relu'))
# Add the output layer with 2 neurons (for binary classification)
model.add(Dense(units=1, activation='softmax'))

# Compile the model (specify loss function, optimizer, and metrics)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print a summary of the model's architecture
model.summary()

# Define the number of epochs and batch size
epochs = 100
batch_size = 8

# Train the model
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

# Make predictions on the test data
y_pred_probabilities = model.predict(X_test)

# Apply a threshold of 0.5 to classify patients
threshold = 0.5
y_pred_binary = (y_pred_probabilities > threshold)

# Print the predicted labels
print("Predicted Labels (Binary):")
print(y_pred_binary)

The above output as be understood as:
1.   **True**: The model's predicted probability for that
     particular patient is greater than 0.5, and the patient is classified as having heart disease.

2.   **False**: The model's predicted probability for
     that particular patient is less than or equal to 0.5, and the patient is classified as not having heart disease.

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Create a train and validation data generator
train_datagen = ImageDataGenerator(
  rescale = 1./255,
  shear_range = 0.1,
  rotation_range = 10,
  zoom_range = 0.1,
  width_shift_range = 0.1,
  height_shift_range = 0.1,
  fill_mode = "nearest",
  validation_split = 0.2
)
val_datagen = ImageDataGenerator(rescale=1./255,
validation_split = 0.2)

#load the digits from Google Drive and split them into train and validation sets.
train_generator = train_datagen.flow_from_directory(
  path + "/Dzo_MNIST",
  target_size = (28, 28),
  batch_size = 32,
  classes = ['0', '1', '2'],
  class_mode = 'categorical',
  color_mode = 'grayscale',
  subset = 'training') # set as training data

validation_generator = val_datagen.flow_from_directory(
  path + "/Dzo_MNIST", # same directory as training data
  target_size = (28, 28),
  batch_size=32,
  classes = ['0', '1', '2'],
  class_mode='categorical',
  color_mode = 'grayscale',
  subset = 'validation') # set as validation data

#Print summary of the data
print("[INFO]- Training Set")
print("Number of samples in train set: ",
train_generator.samples)
print("Number of classes in test set: ",
len(train_generator.class_indices))
print("Number of samples per class[train-set]: ",
int(train_generator.samples /
len(train_generator.class_indices)))
print("****************************************")
print("\n[INFO]- Validation Set")
print("Number of samples in validation set: ",
validation_generator.samples)
print("Number of classes in validation set: ",
len(validation_generator.class_indices))
print("Number of samples per class[validation-set]: ",
int(validation_generator.samples /
len(validation_generator.class_indices)))
print("****************************************")

# Instantiate a Sequential model
model = Sequential()

# First convolution layer with 32 kernels of shape 3 and ReLU activation
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
model.add(Activation("relu"))

# Second convolution layer with 32 kernels of shape 3, ReLU activation,
#max pooling with size 2 by 2, and dropout rate of 0.25
model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Third convolution layer with 64 kernels of shape 3, ReLU activation,
#max pooling with size 2 by 2, dropout rate of 0.25, and a flattened layer
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())

#Fully connected layer with 512 neurons, ReLU activation, and batch normalization
model.add(Dense(512))
model.add(Activation("relu"))
model.add(BatchNormalization())

#Dropout layer with a rate of 0.5
model.add(Dropout(0.5))

#Output layer with 3 neurons and softmax activation
model.add(Dense(3))
model.add(Activation("softmax"))

# Print model summary to see the architecture
model.summary()

#Compile the model
model.compile(
    loss='categorical_crossentropy',  # Categorical cross-entropy loss for multi-class classification
    optimizer='adam',                 # Adam optimizer
    metrics=['accuracy']             # Metric to monitor during training
)

#Fit the neural networks
H = model.fit(
train_generator,
steps_per_epoch = len(train_generator),
validation_data = validation_generator,
validation_steps = len(validation_generator),
epochs = 10
)

# Train accuracy
scores = model.evaluate(train_generator,
steps=len(train_generator), verbose=1)
print("Train Accuracy: %.2f%%" % (scores[1]*100))
# Validation accuracy
scores = model.evaluate(validation_generator,
steps=len(validation_generator), verbose=1)
print("Validation (Seen) Accuracy: %.2f%%" % (scores[1]*100))

# Save the entire model (architecture and weights)
model.save("dzongkha_digit_recognition_model.h5")

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
loaded_model = load_model("dzongkha_digit_recognition_model.h5")

# Load and preprocess the new image for prediction
img_path = "/content/drive/MyDrive/Deep learning/week 10/Sample Handwritten Digit/1630983560.433302.jpg"  # Replace with the path to your new image
img = image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Make predictions
predictions = loaded_model.predict(img_array)

# Interpret the predictions (assuming 3 classes: 0, 1, 2)
predicted_class = np.argmax(predictions, axis=1)

# Define class labels (0, 1, 2)
class_labels = ['0', '1', '2']

# Get the predicted class label
predicted_label = class_labels[predicted_class[0]]

# Print the predicted label
print("Predicted class label:", predicted_label)

import tensorflow.keras.datasets.mnist as mnist

(features_train, label_train), (features_test, label_test) = mnist.load_data()

features_train = features_train.reshape(60000, 28, 28, 1)
features_test = features_test.reshape(10000, 28, 28, 1)

features_train = features_train / 255.0
features_test = features_test / 255.0

# shape of the training set
features_train.shape

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

np.random.seed(8)
tf.random.set_seed(8)

model = tf.keras.Sequential()

conv_layer1 = layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1))
conv_layer2 = layers.Conv2D(64, (3, 3), activation='relu')

# Create a Flatten layer
flatten_layer = Flatten()

# Create a Dense layer with 10 neurons and softmax activation
fc_layer1 = Dense(units=128, activation='softmax')
fc_layer2 = layers.Dense(10, activation='softmax')

model.add(conv_layer1)
model.add(layers.MaxPooling2D(2, 2))
model.add(conv_layer2)
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(fc_layer1)
model.add(fc_layer2)

optimizer = tf.keras.optimizers.Adam(0.001)

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(features_train, label_train, epochs=5, validation_split=0.2, verbose=2)

model.evaluate(features_test, label_test)

import tensorflow.keras.datasets.mnist as mnist

(features_train, label_train), (features_test, label_test) = mnist.load_data()

features_train = features_train.reshape(60000, 28, 28, 1)
features_test = features_test.reshape(10000, 28, 28, 1)

features_train = features_train / 255.0
features_test = features_test / 255.0

# shape of the training set
features_train.shape

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

np.random.seed(8)
tf.random.set_seed(8)

model = tf.keras.Sequential()

conv_layer1 = layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1))
conv_layer2 = layers.Conv2D(64, (3, 3), activation='relu')

# Create a Flatten layer
flatten_layer = Flatten()

# Create a Dense layer with 10 neurons and softmax activation
fc_layer1 = Dense(units=128, activation='softmax')
fc_layer2 = layers.Dense(10, activation='softmax')

model.add(conv_layer1)
model.add(layers.MaxPooling2D(2, 2))
model.add(conv_layer2)
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(fc_layer1)
model.add(fc_layer2)

optimizer = tf.keras.optimizers.Adam(0.001)

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(features_train, label_train, epochs=5, validation_split=0.2, verbose=2)

model.evaluate(features_test, label_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization ,Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np

#Create a train and validation data generator
train_datagen = ImageDataGenerator(
  rescale = 1./255,
  shear_range = 0.1,
  rotation_range = 10,
  zoom_range = 0.1,
  width_shift_range = 0.1,
  height_shift_range = 0.1,
  fill_mode = "nearest",
  validation_split = 0.2
)
val_datagen = ImageDataGenerator(rescale=1./255,
validation_split = 0.2)

#load the digits from Google Drive and split them into train and validation sets.
train_generator = train_datagen.flow_from_directory(
  path + "/Dzo_MNIST",
  target_size = (28, 28),
  batch_size = 32,
  classes = ['0', '1', '2'],
  class_mode = 'categorical',
  color_mode = 'grayscale',
  subset = 'training') # set as training data

validation_generator = val_datagen.flow_from_directory(
  path + "/Dzo_MNIST", # same directory as training data
  target_size = (28, 28),
  batch_size=32,
  classes = ['0', '1', '2'],
  class_mode='categorical',
  color_mode = 'grayscale',
  subset = 'validation') # set as validation data

#Create the neural network architecture
model = Sequential()

# First Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(MaxPooling2D(2, 2))

# Second Convolutional Layer
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(2, 2))

# Flatten Layer to transition from convolutional layers to dense layers
model.add(Flatten())

# Fully Connected Layer with 128 neurons and ReLU activation
model.add(Dense(128, activation='relu'))

# Output Layer with 3 neurons for classification and softmax activation
model.add(Dense(3, activation='softmax'))

#Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

#Fit the model
H = model.fit(
  train_generator,
  steps_per_epoch = len(train_generator),
  validation_data = validation_generator,
  validation_steps = len(validation_generator),
  epochs = 10
)

#Plot the training and validation loss
loss_train = H.history['loss']
loss_val = H.history['val_loss']
epochs = np.arange(1, 10)
plt.plot(loss_train, 'g')
plt.plot(loss_val, 'b')
plt.xticks(np.arange(0, 10, 2))
plt.title('Training and Validation Loss')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend(['train','validation'])
plt.show()

#Plot the training and validation accuracy
acc_train = H.history['accuracy']
acc_val = H.history['val_accuracy']
epochs = np.arange(1, 10)
plt.plot(acc_train, 'g')
plt.plot(acc_val, 'b')
plt.xticks(np.arange(0, 10, 2))
plt.title('Training and Validation Accuracy')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend(['train','validation'])
plt.show()

#Plot the confusion and classification report
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
# Set the figure size
plt.figure(figsize=(4, 4))
predIdxs = model.predict(validation_generator, steps = len(validation_generator))
y_pred = np.argmax(predIdxs, axis = 1)
# Define confusion matrix
# confusion_matrix(y_true, y_pred)
matrix = confusion_matrix(validation_generator.classes, y_pred)
sns.heatmap(matrix, annot=True, cbar=True, fmt='d')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
# Classification Report
print(classification_report(validation_generator.classes, y_pred))

#Retrain the model by adding BatchNormalization after the first Conv2D
# Define the model
model_with_batchnorm = Sequential()

# First Convolutional Layer with BatchNormalization
model_with_batchnorm.add(Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model_with_batchnorm.add(BatchNormalization())  # Add BatchNormalization
model_with_batchnorm.add(MaxPooling2D(2, 2))

# Second Convolutional Layer
model_with_batchnorm.add(Conv2D(64, (3, 3), activation='relu'))
model_with_batchnorm.add(MaxPooling2D(2, 2))

# Flatten Layer
model_with_batchnorm.add(Flatten())

# Fully Connected Layer with 128 neurons and ReLU activation
model_with_batchnorm.add(Dense(128, activation='relu'))

# Output Layer with 3 neurons for classification and softmax activation
model_with_batchnorm.add(Dense(3, activation='softmax'))

#Compile and train the model with BatchNormalization
model_with_batchnorm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
H_with_batchnorm = model_with_batchnorm.fit(
    train_generator,
    steps_per_epoch=len(train_generator),
    validation_data=validation_generator,
    validation_steps=len(validation_generator),
    epochs=10
)

#after training, generate the confusion matrix for both models using the same code as in step 11

# Generate predictions on the validation dataset for the model without BatchNormalization
predIdxs = model.predict(validation_generator, steps=len(validation_generator))
y_pred = np.argmax(predIdxs, axis=1)
true_labels = validation_generator.classes

# Calculate and plot the confusion matrix for the model without BatchNormalization
matrix = confusion_matrix(true_labels, y_pred)
plt.figure(figsize=(4, 4))
sns.heatmap(matrix, annot=True, cbar=True, fmt='d')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix (Without BatchNormalization)')
plt.show()

from tensorflow.keras.datasets import cifar10

(features_train, label_train), (features_test, label_test) = cifar10.load_data()

batch_size = 16
img_height = 32
img_width = 32

from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_img_gen = ImageDataGenerator(
    rescale = 1./255,
    width_shift_range = 0.1,
    height_shift_range = 0.1,
    horizontal_flip = True)

val_img_gen = ImageDataGenerator(rescale=1./255)

train_data_gen = train_img_gen.flow(
    features_train,
    label_train,
    batch_size = batch_size)

val_data_gen = train_img_gen.flow(
    features_test,
    label_test,
    batch_size = batch_size)


import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

np.random.seed(8)
tf.random.set_seed(8)

from keras.api._v2.keras import activations
model = tf.keras.Sequential()
model.add(layers.Conv2D(64, 3, activation='relu', input_shape=(img_height, img_width, 3)))
model.add(layers.MaxPooling2D())
model.add(layers.Conv2D(128, 3, activation='relu'))
model.add(layers.MaxPooling2D())
model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

optimizer = tf.keras.optimizers.Adam(0.001)

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit_generator(
    train_data_gen,
    steps_per_epoch=len(features_train) // batch_size,
    epochs=5,
    validation_data=val_data_gen,
    validation_steps=len(features_test) // batch_size
)

# Write a code to print the shape of features_train, label_train, features_test,
# and label_test and explain the output.

print("shape of features_train:", features_train.shape)
print("shape of label_train:", label_train.shape)
print("shape of features_test:", features_test.shape)
print("shape of label_test:", label_test.shape)

"In the output:
features_train is a numpy array of shape (50000, 32, 32, 3). This means that it contains 50000 images, each of size 
32x32 pixels, and with 3 color channels (RGB). label_train is a numpy array of shape (50000, 1). This means that it contains 
50000 labels, one for each image in features_train. features_test is a numpy array of shape (10000, 32, 32, 3). This means 
that it contains 10000 images, each of size 32x32 pixels, and with 3 color channels (RGB). label_test is a numpy array of 
shape (10000, 1). This means that it contains 10000 labels, one for each image in features_test."

# Explain the output generated by the model.fit_generator.
" The model.fit_generator() function is used to train a neural network using data generators created from the training and 
validation sets. Here's what the output generated by model.fit_generator() means:
The output shows the progress of each epoch during training. It shows the current epoch number, the total number of 
epochs, the number of steps taken in the current epoch, and the total number of steps in the epoch. For example, 
"Epoch 1/5, 3125/3125" means that we are in the first epoch out of 5, and we have taken 3125 steps out of a total of 
3125 steps in the epoch. The output also shows the loss and accuracy of the model on the training set for the current epoch. 
For example, "loss: 2.3029 - accuracy: 0.1002" means that the current loss on the training set is 2.3029 and the current 
accuracy is 0.1002. The output also shows the loss and accuracy of the model on the validation set for the current epoch. 
For example, "val_loss: 2.3028 - val_accuracy: 0.0858" means that the current loss on the validation set is 2.3028 and the 
current accuracy is 0.0858. The output continues for each epoch until all epochs are completed. " 

# Modify the ImageDataGenerator function by adding vertical_flip, shear_range,
# rotation_range and brightness_range and fit the model with the same network
# design. Compare the result with the first model.

train_datagen = ImageDataGenerator(
    rescale=1./255,
    vertical_flip=True,
    shear_range=0.2,
    rotation_range=20,
    brightness_range=[0.2, 1.0]
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(
    features_train,
    label_train,
    batch_size=32
)

test_generator = test_datagen.flow(
    features_test,
    label_test,
    batch_size=32
)

# train the model with the new data generators
history = model.fit_generator(
    train_generator,
    steps_per_epoch=len(features_train) / 32,
    epochs=5,
    validation_data=test_generator,
    validation_steps=len(features_test) / 32
)


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


import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('heart.csv')
X = df.drop('target',axis=1)

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)

model = Sequential()
model.add(Dense(13,input_dim=13,activation='relu'))
model.add(Dense(8,activation='relu'))
model.add(Dense(12,activation='relu'))
model.add(Dense(14,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

history = model.fit(X_train,y_train,epochs = 100,batch_size = 8,validation_data = (X_test,y_test))

loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = np.arange(1,100)
plt.plot(loss_train,'g')
plt.plot(loss_val,'b')
plt.xticks(np.arange(0,100,5))
plt.title('Training and Validation Loss')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend(['train','validation'])
plt.show()

acc_train = history.history['accuracy']
acc_val = history.history['val_accuracy']
epochs = np.arange(1,100)
plt.plot(acc_train,'g')
plt.plot(acc_val,'b')
plt.xticks(np.arange(0,100,5))
plt.title('Training and validation accuracy')
plt.xlabel('Number of Epochs')
plt.ylabel('Loss')
plt.legend(['train','validation'])
plt.show()

import tensorflow.keras.datasets.mnist as mnist

(features_train, label_train), (features_test, label_test) = mnist.load_data()

features_train = features_train.reshape(60000, 28, 28, 1)
features_test = features_test.reshape(10000, 28, 28, 1)

features_train = features_train / 255.0
features_test = features_test / 255.0

# shape of the training set
features_train.shape

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

np.random.seed(8)
tf.random.set_seed(8)

model = tf.keras.Sequential()

conv_layer1 = layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1))
conv_layer2 = layers.Conv2D(64, (3, 3), activation='relu')

# Create a Flatten layer
flatten_layer = Flatten()

# Create a Dense layer with 10 neurons and softmax activation
fc_layer1 = Dense(units=128, activation='softmax')
fc_layer2 = layers.Dense(10, activation='softmax')

model.add(conv_layer1)
model.add(layers.MaxPooling2D(2, 2))
model.add(conv_layer2)
model.add(layers.MaxPooling2D(2, 2))
model.add(layers.Flatten())
model.add(fc_layer1)
model.add(fc_layer2)

optimizer = tf.keras.optimizers.Adam(0.001)

model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

model.fit(features_train, label_train, epochs=5, validation_split=0.2, verbose=2)

model.evaluate(features_test, label_test)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#Create a train and validation data generator
train_datagen = ImageDataGenerator(
  rescale = 1./255,
  shear_range = 0.1,
  rotation_range = 10,
  zoom_range = 0.1,
  width_shift_range = 0.1,
  height_shift_range = 0.1,
  fill_mode = "nearest",
  validation_split = 0.2
)
val_datagen = ImageDataGenerator(rescale=1./255,
validation_split = 0.2)

#load the digits from Google Drive and split them into train and validation sets.
train_generator = train_datagen.flow_from_directory(
  path + "/Dzo_MNIST",
  target_size = (28, 28),
  batch_size = 32,
  classes = ['0', '1', '2'],
  class_mode = 'categorical',
  color_mode = 'grayscale',
  subset = 'training') # set as training data

validation_generator = val_datagen.flow_from_directory(
  path + "/Dzo_MNIST", # same directory as training data
  target_size = (28, 28),
  batch_size=32,
  classes = ['0', '1', '2'],
  class_mode='categorical',
  color_mode = 'grayscale',
  subset = 'validation') # set as validation data

#Print summary of the data
print("[INFO]- Training Set")
print("Number of samples in train set: ",
train_generator.samples)
print("Number of classes in test set: ",
len(train_generator.class_indices))
print("Number of samples per class[train-set]: ",
int(train_generator.samples /
len(train_generator.class_indices)))
print("****************************************")
print("\n[INFO]- Validation Set")
print("Number of samples in validation set: ",
validation_generator.samples)
print("Number of classes in validation set: ",
len(validation_generator.class_indices))
print("Number of samples per class[validation-set]: ",
int(validation_generator.samples /
len(validation_generator.class_indices)))
print("****************************************")

# Instantiate a Sequential model
model = Sequential()

# First convolution layer with 32 kernels of shape 3 and ReLU activation
model.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
model.add(Activation("relu"))

# Second convolution layer with 32 kernels of shape 3, ReLU activation,
#max pooling with size 2 by 2, and dropout rate of 0.25
model.add(Conv2D(32, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Third convolution layer with 64 kernels of shape 3, ReLU activation,
#max pooling with size 2 by 2, dropout rate of 0.25, and a flattened layer
model.add(Conv2D(64, (3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())

#Fully connected layer with 512 neurons, ReLU activation, and batch normalization
model.add(Dense(512))
model.add(Activation("relu"))
model.add(BatchNormalization())

#Dropout layer with a rate of 0.5
model.add(Dropout(0.5))

#Output layer with 3 neurons and softmax activation
model.add(Dense(3))
model.add(Activation("softmax"))

# Print model summary to see the architecture
model.summary()

#Compile the model
model.compile(
    loss='categorical_crossentropy',  # Categorical cross-entropy loss for multi-class classification
    optimizer='adam',                 # Adam optimizer
    metrics=['accuracy']             # Metric to monitor during training
)

#Fit the neural networks
H = model.fit(
train_generator,
steps_per_epoch = len(train_generator),
validation_data = validation_generator,
validation_steps = len(validation_generator),
epochs = 10
)

# Train accuracy
scores = model.evaluate(train_generator,
steps=len(train_generator), verbose=1)
print("Train Accuracy: %.2f%%" % (scores[1]*100))
# Validation accuracy
scores = model.evaluate(validation_generator,
steps=len(validation_generator), verbose=1)
print("Validation (Seen) Accuracy: %.2f%%" % (scores[1]*100))

# Save the entire model (architecture and weights)
model.save("dzongkha_digit_recognition_model.h5")

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
loaded_model = load_model("dzongkha_digit_recognition_model.h5")

# Load and preprocess the new image for prediction
img_path = "/content/drive/MyDrive/Deep learning/week 10/Sample Handwritten Digit/1630983560.433302.jpg"  # Replace with the path to your new image
img = image.load_img(img_path, target_size=(28, 28), color_mode="grayscale")
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Make predictions
predictions = loaded_model.predict(img_array)

# Interpret the predictions (assuming 3 classes: 0, 1, 2)
predicted_class = np.argmax(predictions, axis=1)

# Define class labels (0, 1, 2)
class_labels = ['0', '1', '2']

# Get the predicted class label
predicted_label = class_labels[predicted_class[0]]

# Print the predicted label
print("Predicted class label:", predicted_label)
