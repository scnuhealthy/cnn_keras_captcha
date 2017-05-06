#!flask/bin/python
from flask import Flask
from flask import request
from eval_test import CaptchaEval
import io
from PIL import Image


app = Flask(__name__)


captchaEval = CaptchaEval()


@app.route('/captcha', methods=['POST'])
def index():

    fileContent = request.files['file']
    stream = io.BytesIO(fileContent.read())

    localImage = Image.open(fileContent)

    text = captchaEval.predict_from_img(localImage)

    print (text)
    return text

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False)


#curl -i -X POST -F file=@/tmp/resized.png  http://127.0.0.1:5000/captcha
