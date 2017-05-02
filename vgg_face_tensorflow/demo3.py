# written digit recognition
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


mnist_data = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)
img_load = np.loadtxt('number_image.txt')
print 'image_sample_size:', img_load.shape

img = mnist_data.test.images[1]
img = img_load
#imgm = np.mean(mnist_data.train.images, axis=0)


with open('mnist.tfmodel', mode='rb') as f:
	fileContent = f.read()
graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
images = tf.placeholder(tf.float32, [None, 28, 28, 1])
tf.import_graph_def(graph_def, input_map={'images':images, 'Deepnn/keep_prob':1.0})
graph = tf.get_default_graph()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
  	print "variables initialized"
  	batch = img.reshape((1, 28, 28, 1))
  	prob_tensor = graph.get_tensor_by_name('import/Deepnn/prob:0')
  	prob, prob_ = sess.run([tf.nn.softmax(prob_tensor), prob_tensor], feed_dict={images:batch})

print 'softmax prob:', prob
print 'original prob:', prob_
plt.figure()
plt.imshow(img.reshape(28,28), cmap='gray')
plt.show()

