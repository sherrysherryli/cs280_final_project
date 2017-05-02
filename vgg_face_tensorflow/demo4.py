# reshape, blur, sharpen and turn color images to grayscale
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.ndimage as ndimg
import scipy.misc as misc
import math
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data


def read_reshape_blur_image(url, pixel, sigma):
	image = Image.open(url)
	large = pixel*max(image.size)/min(image.size)
	tt = (large-pixel)/2
	if np.argmin(image.size) == 0:
		reshaped_image = np.asarray(image.resize((pixel, large)))
		reshaped_image = reshaped_image[tt:tt+pixel,:,:]
	else:
		reshaped_image = np.asarray(image.resize((large, pixel)))
		reshaped_image = reshaped_image[:,tt:tt+pixel,:]
	return ndimg.gaussian_filter(reshaped_image/255., (sigma,sigma,0))

def sharpen_gray_image(img, alpha):
	img = np.dot(img[...,:3], [0.299, 0.587, 0.114])
	filter_blurred_img = ndimg.gaussian_filter(img, 1)
	return img + alpha * (img - filter_blurred_img)


# load mnist data set
mnist_data = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)
mnist_ref = np.mean(mnist_data.train.images, axis=0)
mnist_ref = mnist_ref.reshape((1,28,28,1))
face_ref0 = read_reshape_blur_image('average_face.jpg', 224, 5)
face_ref = face_ref0.reshape((1,224,224,3))

face_ref_sharpen = sharpen_gray_image(face_ref0, 50)

f = plt.figure()
f.add_subplot(1,2,1)
plt.imshow(face_ref0)
f.add_subplot(1,2,2)
plt.imshow(face_ref_sharpen, cmap='gray')
plt.show()
