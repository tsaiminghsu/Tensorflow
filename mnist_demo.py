# -*- coding: utf-8 -*-
"""
Created on Sat Jun  8 17:38:58 2019

@author: Attlie
"""
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,Conv2D,MaxPooling2D
from keras.optimizers import RMSprop
import numpy as np
import matplotlib.pyplot as plt

epochs = 10
np.random.seed(10)
(x_Train, y_Train), (x_Test, y_Test) = mnist.load_data()
x_Train4D=x_Train.reshape(x_Train.shape[0],28,28,1).astype('float32')
x_Test4D=x_Test.reshape(x_Test.shape[0],28,28,1).astype('float32')
x_Train4D_normalize = x_Train4D / 255
x_Test4D_normalize = x_Test4D / 255
y_TrainOneHot = np_utils.to_categorical(y_Train)
y_TestOneHot = np_utils.to_categorical(y_Test)
model = Sequential()
model.add(Conv2D(filters=16,
                 kernel_size=(5,5),
                 padding='same',
                 input_shape=(28,28,1), 
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(filters=36,
                 kernel_size=(5,5),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
#######
model.add(Dense(512, activation='relu',input_shape=(784,)))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10,activation='softmax'))
print(model.summary())
model.compile(loss='categorical_crossentropy',
#              optimizer=RMSprop(),metrics=['accuracy'])  #best_epochs = 5 - acc:0.99
              optimizer='adam',metrics=['accuracy'])   #best_epochs = 5 - acc:0.99
train_history=model.fit(x=x_Train4D_normalize, y=y_TrainOneHot,
                        validation_split=0.2, 
                        epochs=epochs, batch_size=300,verbose=2)
def show_train_history(train_history,train_acc,test_acc):
    plt.plot(train_history.history[train_acc])
    plt.plot(train_history.history[test_acc])
    plt.title('Train History')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
#show_train_history(train_history,'acc','val_acc')
#show_train_history(train_history,'loss','val_loss')       
score = model.evaluate(x=x_Train4D_normalize, y=y_TrainOneHot)
score[1]
prediction=model.predict_classes(x_Test4D_normalize)
def plot_images_labels_prediction(images,labels,prediction,idx,num=10):
    fig = plt.gcf()
    fig.set_size_inches(12, 14)
    if num>25: num=25 
    for i in range(0, num):
        ax=plt.subplot(5,5, 1+i)
        ax.imshow(images[idx], cmap='binary')

        ax.set_title("label=" +str(labels[idx])+
                     ",predict="+str(prediction[idx])
                     ,fontsize=10) 
        
        ax.set_xticks([]);ax.set_yticks([])        
        idx+=1 
    plt.show()
 
#plot_images_labels_prediction(x_Test,y_Test,prediction,idx=0)
