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
