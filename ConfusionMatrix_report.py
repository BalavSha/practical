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

