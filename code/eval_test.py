
import sys
from load_data import *
import captcha_params
import load_model

# input image dimensions
img_rows, img_cols = captcha_params.get_height(), captcha_params.get_width()


MAX_CAPTCHA = captcha_params.get_captcha_size()
CHAR_SET_LEN = captcha_params.get_char_set_len()


# the data, shuffled and split between train and test sets
img = sys.argv[1]

X_test = get_x_input_from_file(img)

# load the trained model
model = load_model.get_model('')

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

predict = model.predict(X_test)


text = ''
for i in range(X_test.shape[0]):
    true = []
    predict2 = []
    for j in range(MAX_CAPTCHA):
        char_index = get_max(predict[i,CHAR_SET_LEN*j:(j+1)*CHAR_SET_LEN])
        char = captcha_params.get_char_set()[char_index]
        predict2.append(char)
    text = text.join(predict2)

print(text)
