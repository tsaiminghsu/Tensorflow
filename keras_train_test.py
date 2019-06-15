# -*- coding: utf-8 -*-
"""
Created on Tue May 28 14:15:12 2019

@author: USER
"""

import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense , Dropout , Flatten
from keras.layers import Conv2D , MaxPooling2D

x_train = np.arange(0 , 1 , 0.01)
y_train = np.sin(2 * np.pi * x_train)
plt.plot(x_train,y_train,linestyle='-',label = 'Ground truth')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.show()

batch_szie = 1
epochs = 1000
model = Sequential()
model.add(Dense(5,activation='tanh',input_shape=(1,)))
model.add(Dense(5,activation='tanh'))
model.add(Dense(1,activation='linear'))

model.compile(loss='mean_squared_error',optimizer='sgd',metrics=['mean_squared_error'])
model.fit(x_train,y_train,
          batch_size=batch_szie,
          epochs=epochs,
          verbose=1)
x_test = np.copy(x_train)
y_pred = model.predict(x_test)

plt.plot(x_test , y_pred , linestyle='-.',label='Prediction')
plt.legend()
plt.show()
