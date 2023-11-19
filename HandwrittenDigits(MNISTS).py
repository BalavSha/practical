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
