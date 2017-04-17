# Read image
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
#import skimage
#import skimage.io
#import skimage.transform

# read image using tensorflow
filename = tf.train.string_input_producer(["ak.png"]) 
reader = tf.WholeFileReader()
a, image_file = reader.read(filename)
image_tensor = tf.image.decode_png(image_file)
init = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord)
	image = image_tensor.eval()
  	print(image)
  	coord.request_stop()
  	coord.join(threads)

# read image using matplotlib
image2 = mpimg.imread("ak.png")*255
image2 = image2.astype(int)
print(image2[0][0][0])