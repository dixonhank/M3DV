import os
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense,Activation,Convolution3D,MaxPooling3D,Dropout,Flatten,BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras import utils,optimizers,initializers,regularizers
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint
from tensorflow.keras import backend as k
from sklearn.model_selection import train_test_split
k.set_image_data_format('channels_last')
import numpy as np
import csv
from dataloader import data_loader

x,y,z=32,32,32

batch_size=8
nb_epoch=100

X_test=data_loader(x,y,z,'test')
X_train=data_loader(x,y,z,'train')
y_train=data_loader(x,y,z,'label')
Y_train=utils.to_categorical(y_train,2)
index=[i for i in range(np.size(X_train,0))] #data enhancement with data shuffle
random.shuffle(index)
X_train=X_train[index,:,:,:]
Y_train=Y_train[index]

#X_train_s,X_val,Y_train_s,Y_val=train_test_split(X_train,Y_train,test_size=0.3,random_state=3)

np.set_printoptions(threshold=100000000)

model1=Sequential()

'''layer1'''
model1.add(Convolution3D(
    16,
    kernel_size=(3,3,3),
    input_shape=(x,y,z,1)
))

model1.add(BatchNormalization(axis=4))

model1.add(Activation('relu'))

model1.add(MaxPooling3D(
    pool_size=(2,2,2),
    strides=(2,2,2)
))


'''layer2'''
model1.add(Convolution3D(
    32,
    kernel_size=(3,3,3),
    kernel_initializer='random_uniform'
))

model1.add(BatchNormalization(axis=4))

model1.add(Activation('relu'))

model1.add(MaxPooling3D(
    pool_size=(2,2,2),
    strides=(2,2,2)
))


'''layer3'''
model1.add(Convolution3D(
    64,
    kernel_size=(3,3,3),
    padding='same',
    kernel_initializer='random_uniform'
))

model1.add(BatchNormalization(axis=4))

model1.add(Activation('relu'))

model1.add(Convolution3D(
    64,
    kernel_size=(3,3,3),
    padding='same',
    kernel_initializer='random_uniform'
))

model1.add(BatchNormalization(axis=4))

model1.add(Activation('relu'))

model1.add(MaxPooling3D(
    pool_size=(2,2,2),
    strides=(2,2,2)
))


'''layer4'''
model1.add(Convolution3D(
    128,
    kernel_size=(3,3,3),
    activation='relu',
    kernel_initializer='random_uniform'
))

model1.add(Flatten())
model1.add(Dropout(0.5))
model1.add(Dense(64,activation='relu'))
model1.add(Dropout(0.5))
model1.add(Dense(2))
model1.add(Activation('softmax'))

model1.summary()

# Compile
tf.keras.optimizers.RMSprop(lr=0.001)
model1.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics=['accuracy'])

#early_stopping=EarlyStopping(monitor='val_loss',min_delta=0,mode='min',patience=10)
checkpoint=ModelCheckpoint('new.h5',
                           monitor='val_loss',
                           save_best_only=True,
                           mode='min',
                           period=1
                           )

hist=model1.fit(
    X_train,
    Y_train,
    validation_split=0.3,
    batch_size=batch_size,
    nb_epoch=nb_epoch,
    shuffle=True,
    callbacks=[checkpoint]
    )

#score1=model1.evaluate(
#    X_val,
#    Y_val,
#    batch_size=batch_size,
#    )
#
#train_loss=hist.history['loss']
#val_loss=hist.history['val_loss']
#train_acc=hist.history['acc']
#val_acc=hist.history['val_acc']
#print('**********************************************')
#print('Test score:', score1)
#print('History', hist.history)
#print('train_loss', train_loss)
#print('val_loss', val_loss)
#print('train_acc', train_acc)
#print('val_acc', val_acc)

pre1=model1.predict_proba(X_test)
np.savetxt('new.csv',pre1[:,1])
