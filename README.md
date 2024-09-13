# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

## Neural Network Model

Include the neural network model diagram.

## DESIGN STEPS

- **Step 1:** Import tensorflow and preprocessing libraries
- **Step 2:** Download and load the dataset
- **Step 3:** Scale the dataset between it's min and max values
- **Step 4:** Using one hot encode, encode the categorical values
- **Step 5:** Split the data into train and test
- **Step 6:** Build the convolutional neural network model
- **Step 7:** Train the model with the training data
- **Step 8:** Plot the performance plot
- **Step 9:** Evaluate the model with the testing data
- **Step 10:** Fit the model and predict the single input


## PROGRAM

### Name: ROSHINI R K
### Register Number: 21222230123

```
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train.shape
X_test.shape
single_image = X_train[0]
single_image.shape
plt.imshow(single_image,cmap='gray')
y_train.shape

X_train.min()
X_train.max()
X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0
X_train_scaled.min()
X_train_scaled.max()

y_train[0]
y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)
type(y_train_onehot)
y_train_onehot.shape
single_image = X_train[500]
plt.imshow(single_image,cmap='gray')
y_train_onehot[500]

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

model = keras.Sequential()
model.add(layers.Input(shape=(28,28,1)))
model.add(layers.Conv2D(filters=16,kernel_size=(3,3),activation='relu'))
model.add(layers.Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(units=32,activation='relu'))
model.add(layers.Dense(units=10,activation='softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics='accuracy')

model.summary()

model.fit(X_train_scaled,y_train_onehot,epochs=5,batch_size=64,validation_data=(X_test_scaled,y_test_onehot))

metrics = pd.DataFrame(model.history.history)
metrics.head()

metrics[['accuracy','val_accuracy']].plot()
metrics[['loss','val_loss']].plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled),axis=1)

print(confusion_matrix(y_test,x_test_predictions))
print(classification_report(y_test,x_test_predictions))

img = image.load_img('imagefive.jpg')
type(img)
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_scaled = img_28_gray.numpy()/255.0
x_single_prediction = np.argmax(model.predict(img_scaled.reshape(1,28,28,1)),axis=1)
print(x_single_prediction)
plt.imshow(img_scaled.reshape(28,28),cmap='gray')
img_inv = 255.0 - img_28_gray
img_inv_scaled = img_inv.numpy()/255.0
print(x_single_prediction)
```



## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://github.com/user-attachments/assets/b16753b6-e9e0-4f04-a292-c1038042447f)


### Classification Report
![image](https://github.com/user-attachments/assets/cf9c4a6d-a53f-4093-af1f-7cf851b4a943)





### Confusion Matrix

![image](https://github.com/user-attachments/assets/cc85e2ce-5a4c-49af-b795-18921a81fddc)


### New Sample Data Prediction

![image](https://github.com/user-attachments/assets/25d1fb21-572d-4090-9fea-02eaec1c1f6f)


## RESULT
Thus a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully
