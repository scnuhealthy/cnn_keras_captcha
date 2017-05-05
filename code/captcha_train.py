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

from keras.models import Sequential
from keras.utils import np_utils
from keras import backend as K
from load_data import *
import h5py
from keras.callbacks import ModelCheckpoint

import captcha_params
import load_model

# input image dimensions
img_rows, img_cols = captcha_params.get_height(), captcha_params.get_width()

batch_size = 128
nb_epoch = 64

MAX_CAPTCHA = captcha_params.get_captcha_size()
CHAR_SET_LEN = captcha_params.get_char_set_len()



# the data, shuffled and split between train and test sets
(X_train, Y_train), (X_test, Y_test) = load_data(tol_num = 24000,train_num = 18000)

# i use the theano backend
if K.image_dim_ordering() == 'th':
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

model = load_model.get_model(input_shape)

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

filepath = load_model.get_weights_file()
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_data=(X_test,Y_test), callbacks=callbacks_list)

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

