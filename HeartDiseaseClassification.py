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
