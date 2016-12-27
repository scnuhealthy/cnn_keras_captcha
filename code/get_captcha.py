from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import os

number = ['0','1','2','3','4','5','6','7','8','9']
alphabet = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
ALPHABET = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']

# generate  the captcha text randomly from the char lists above
def random_captcha_text(char_set=number, captcha_size=4):
	captcha_text = []
	for i in range(captcha_size):
		c = random.choice(char_set)
		captcha_text.append(c)
	return captcha_text
 
# generate the captcha text and image and save the image 
def gen_captcha_text_and_image(i):
	image = ImageCaptcha()
 
	captcha_text = random_captcha_text()
	captcha_text = ''.join(captcha_text)

        path = 'F:/captcha//data//'
        if os.path.exists(path) == False: # if the folder is not existed, create it
                os.mkdir(path)
                
	captcha = image.generate(captcha_text)

	# naming rules: num(in order)+'_'+'captcha text'.include num is for avoiding the same name
	image.write(captcha_text, path+str(i)+'_'+captcha_text + '.jpg') 
 
	captcha_image = Image.open(captcha)
	captcha_image = np.array(captcha_image)
	return captcha_text, captcha_image
 
if __name__ == '__main__':

        
        for i in range(21):     
                text, image = gen_captcha_text_and_image(i)

        # show the image
        '''
	f = plt.figure()
	ax = f.add_subplot(111)
	ax.text(0.1, 0.9,text, ha='center', va='center', transform=ax.transAxes)
	plt.imshow(image)
 
	plt.show()
        '''
