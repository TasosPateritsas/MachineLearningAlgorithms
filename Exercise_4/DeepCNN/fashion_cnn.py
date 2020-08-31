from __future__ import absolute_import, division, print_function, unicode_literals

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, Flatten, 
                                     Dense, Dropout, BatchNormalization)
from tensorflow.keras.models import Sequential
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Scale these values to a range of 0 to 1 before feeding them to the neural network model. 
# To do so, divide the values by 255. 
# It's important that the training set and the testing set be preprocessed in the same way
train_images = train_images / 255.0

test_images = test_images / 255.0
# reshape to mum_train_images X height X width X channels, where channels = 1
train_images_reshaped = np.reshape(train_images, (train_images.shape[0], train_images.shape[1], train_images.shape[2], 1))
test_images_reshaped = np.reshape(test_images, (test_images.shape[0], test_images.shape[1], test_images.shape[2], 1))


IMAGE_HEIGHT = 28
IMAGE_WIDTH =28 
NUM_CHANNELS =1
chanDim =1
# Build the model
# Building the neural network requires configuring the layers of the model, then compiling the model.

model =keras.Sequential() # fill the model
#1
model.add(Conv2D(32, kernel_size=(3, 3), strides =1, activation='relu', padding='same', 
               input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CHANNELS)))
model.add(BatchNormalization(axis=chanDim))

#2
model.add(Conv2D(32, kernel_size=(3, 3), strides =1, activation='relu', padding='same'))
model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
model.add(Dropout(0.2))
#3
model.add(Conv2D(64, kernel_size=(3, 3), strides =1, activation='relu', padding='same'))
model.add(BatchNormalization(axis=chanDim))
#4
model.add(Conv2D(64, kernel_size=(3, 3), strides =1, activation='relu', padding='same'))
model.add(BatchNormalization(axis=chanDim))

model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
model.add(Dropout(0.3))
#5
model.add(Conv2D(128, kernel_size=(3, 3), strides =1, activation='relu', padding='same'))
model.add(BatchNormalization(axis=chanDim))
model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
model.add(Dropout(0.4))

model.add(Flatten())
model.add(Dense(200,activation='relu'))
model.add(BatchNormalization())

model.add(Dropout(0.5))
model.add(Dense(10,activation='relu'))




model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

#Train the model
# Training the neural network model requires the following steps:

#   1. Feed the training data to the model. In this example, the training data is in the train_images and train_labels arrays.
#   2. The model learns to associate images and labels.
#   3. You ask the model to make predictions about a test set—in this example, the test_images array.
#   4. Verify that the predictions match the labels from the test_labels array.

m = model.fit(train_images_reshaped, train_labels, epochs=50, validation_data=(test_images_reshaped, test_labels))

# Evaluate accuracy
test_loss, test_acc = model.evaluate(test_images_reshaped,  test_labels, verbose=2)

print('\nTest accuracy:', test_acc)

# Make predictions
# With the model trained, you can use it to make predictions about some images. 
# The model's linear outputs, logits. 
# Attach a softmax layer to convert the logits to probabilities, which are easier to interpret. 
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

predictions = probability_model.predict(test_images_reshaped)

plot_some_predictions(test_images, test_labels, predictions, class_names, num_rows=5, num_cols=3)

plt.plot(m.history['accuracy'], label='accuracy')
plt.plot(m.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()



