'''
Author:kemo

Trains a captcha datasets, each captcha includes four number.
Gets to 63.9% test accuracy after 64 epochs
(there is still a lot of margin for parameter tuning).
120 seconds per epoch on a Nvidia GeForce 940M GPU.
'''

from __future__ import print_function
import numpy as np
np.random.seed(1337)  # for reproducibility

import captcha_params

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from load_data import *
import h5py
from keras.models import model_from_json

batch_size = 128
nb_epoch = 64

MAX_CAPTCHA = captcha_params.get_captcha_size()
CHAR_SET_LEN = captcha_params.get_char_set_len()

# input image dimensions
img_rows, img_cols = captcha_params.get_height(), captcha_params.get_width()

# number of convolutional filters to use
nb_filters1 = 32
nb_filters2 = 64
nb_filters3 = 64
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

# the data, shuffled and split between train and test sets
(X_train, Y_train), (X_test, Y_test) = load_data(tol_num = 24000,train_num = 18000)

# i use the theano backend
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols,1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')


model = Sequential()


# 3 conv layer
model.add(Convolution2D(nb_filters1, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Convolution2D(nb_filters2, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

model.add(Convolution2D(nb_filters3, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pool_size))
model.add(Dropout(0.25))

# Fully connected layer
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(MAX_CAPTCHA*CHAR_SET_LEN))
model.add(Activation('softmax'))


# load the trained model
# model = model_from_json(open('my_model.json').read())  
# model.load_weights('my_model_weights.h5')

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
            verbose=1, validation_data=(X_test,Y_test))

score = model.evaluate(X_test, Y_test, verbose=0)
predict = model.predict(X_test,batch_size = batch_size,verbose = 0)

# calculate the accuracy with the test data
acc = 0
for i in range(X_test.shape[0]):
    true = []
    predict2 = []
    for j in range(MAX_CAPTCHA):
        true.append(get_max(Y_test[i,CHAR_SET_LEN*j:(j+1)*CHAR_SET_LEN]))
        predict2.append(get_max(predict[i,CHAR_SET_LEN*j:(j+1)*CHAR_SET_LEN]))
    if true == predict2:
        acc+=1
    if i<20:
        print (i,' true: ',true)
        print (i,' predict: ',predict2)
print('predict correctly: ',acc)
print('total prediction: ',X_test.shape[0])
print('Score: ',score)

# save model
json_string = model.to_json()
open("my_model.json","w").write(json_string)
model.save_weights('my_model_weights.h5')
