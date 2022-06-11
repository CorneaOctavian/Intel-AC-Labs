



from skimage.draw import random_shapes
from skimage.color import rgb2gray
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import tensorflow as tf
from tensorflow import keras
from keras.datasets import mnist






result = random_shapes((50, 50), 
              shape = 'rectangle',
              max_shapes = 1,
              intensity_range = ((0, 255), ))

plt.imshow(rgb2gray(result[0]))

rect = []
for i in range(100):
    result = random_shapes((50, 50), 
              shape = 'rectangle',
              max_shapes = 1,
              min_size = 25,
              intensity_range = ((0, 0), ))
    result = rgb2gray(result[0])
    rect.append(result)
    
plt.imshow(rect[2])


circle = []
for i in range(100):
    result = random_shapes((50, 50), 
              shape = 'circle',
              max_shapes = 1,
              intensity_range = ((0, 0), ), 
              min_size = 25)
    result = rgb2gray(result[0])
    circle.append(result)

plt.imshow(circle[3])


triangle = []
for i in range(100):
    result = random_shapes((50, 50), 
              shape = 'triangle',
              max_shapes = 1,
              intensity_range = ((0, 0), ), 
              min_size = 25)
    result = rgb2gray(result[0])
    triangle.append(result)


plt.imshow(triangle[1])





## 0 --> Rectangle
## 1 --> Circle
## 2 --> Triangle

t = np.asarray(triangle)
r = np.asarray(rect)
c = np.asarray(circle)
data = np.concatenate((r, c, t), axis = 0)
plt.imshow(data[200])




labels = [0 for i in range(100)]
labels.extend([1 for i in range(100)])
labels.extend([2 for i in range(100)])
labels[200]



y = to_categorical(labels)
X = data.reshape((3000, 50 * 50))
model = keras.Sequential()
model.add(keras.layers.Dense(500, activation = 'relu',
                             input_shape = (2500, )))
model.add(keras.layers.Dense(3, activation = 'softmax'))
model.compile(optimizer = keras.optimizers.SGD(lr = 0.1),
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])


model.fit(X, y, epochs = 10, batch_size = 10)


(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train[0]

data[0]




