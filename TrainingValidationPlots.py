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

