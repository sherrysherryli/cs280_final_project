# face detection
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import cv2


def reshape_image(url, pixel):
	image = Image.open(url)
	large = pixel*max(image.size)/min(image.size)
	tt = (large-pixel)/2
	if np.argmin(image.size) == 0:
		reshaped_image = np.asarray(image.resize((pixel, large)))
		reshaped_image = reshaped_image[tt:tt+pixel,:,:]
	else:
		reshaped_image = np.asarray(image.resize((large, pixel)))
		reshaped_image = reshaped_image[:,tt:tt+pixel,:]
	return reshaped_image/255.
	
img = reshape_image('ave_face/white_man.jpg', 224)
img = (img*255).astype(np.uint8)[:,:,::-1]


# load model
with open('facedetect.tfmodel', mode='rb') as f:
	fileContent = f.read()
graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
images = tf.placeholder(tf.float32, [None, 224, 224, 3])
tf.import_graph_def(graph_def, input_map={'images:0': images, 'Deepnn/keep_prob': 1.0})
graph = tf.get_default_graph()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
  	print "variables initialized"
  	batch = img.reshape((1, 224, 224, 3))
  	prob_tensor = graph.get_tensor_by_name('import/Deepnn/prob:0')
  	prob, value = sess.run([tf.nn.sigmoid(prob_tensor-tf.constant(1000.)), prob_tensor], feed_dict={images: batch})

print img.shape 
print "Prob of human:", prob[0][0], "Value:", value[0][0]
f = plt.figure()
plt.imshow(img[:,:,::-1])
plt.show()
