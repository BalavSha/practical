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
