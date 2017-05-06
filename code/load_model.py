import os.path
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.models import model_from_json
import captcha_params


MAX_CAPTCHA = captcha_params.get_captcha_size()
CHAR_SET_LEN = captcha_params.get_char_set_len()

# number of convolutional filters to use
nb_filters1 = 32
nb_filters2 = 64
nb_filters3 = 64
# size of pooling area for max pooling
pool_size = (2, 2)
# convolution kernel size
kernel_size = (3, 3)

WEIGHT_FILE = "my_model_weights.h5"

def get_model(input_shape):
    model = Sequential()
    if os.path.exists(WEIGHT_FILE):
        print ("loading the trained model")
        model = model_from_json(open('my_model.json').read())  
        model.load_weights(WEIGHT_FILE)

    else:
        # 3 conv layer
        model.add(Conv2D(nb_filters1, (kernel_size[0], kernel_size[1]), padding='valid', input_shape=input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

        model.add(Conv2D(nb_filters2, (kernel_size[0], kernel_size[1])))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

        model.add(Conv2D(nb_filters3, (kernel_size[0], kernel_size[1])))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=pool_size))
        model.add(Dropout(0.25))

        # Fully connected layer
        model.add(Flatten())
        model.add(Dense(1024*MAX_CAPTCHA))
        model.add(Dense(512*MAX_CAPTCHA))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Dense(MAX_CAPTCHA*CHAR_SET_LEN))
        model.add(Activation('softmax'))
        
    json_string = model.to_json()
    open("my_model.json","w").write(json_string)
    return model


def get_weights_file():
    return WEIGHT_FILE


