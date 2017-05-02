# face recognition
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image


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
	
img = reshape_image('face_image.jpg', 224)
print img.shape

# load model
with open('vggface16.tfmodel', mode='rb') as f:
	fileContent = f.read()
graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
images = tf.placeholder(tf.float32, [None, 224, 224, 3])
tf.import_graph_def(graph_def, input_map={ 'images:0': images })
graph = tf.get_default_graph()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
  	print "variables initialized"
  	batch = img.reshape((1, 224, 224, 3))
  	feed_dict = { images: batch }
  	prob_tensor = graph.get_tensor_by_name('import/prob:0')
  	fc8_tensor = graph.get_tensor_by_name('import/fc8/BiasAdd:0')
  	prob, fc8 = sess.run([prob_tensor,fc8_tensor], feed_dict=feed_dict)
 
name = [s.strip() for s in open('names.txt').readlines()]
rank = np.argsort(prob[0])[::-1]
print fc8.shape
print "top 1:", name[rank[0]], prob[0][rank[0]], fc8[0][rank[0]]
print "top 2:", name[rank[1]], prob[0][rank[1]], fc8[0][rank[1]]
print "top 3:", name[rank[2]], prob[0][rank[2]], fc8[0][rank[2]]
print "top 4:", name[rank[3]], prob[0][rank[3]], fc8[0][rank[3]]
print "top 5:", name[rank[4]], prob[0][rank[4]], fc8[0][rank[4]]

#print graph.get_operations()
plt.figure()
plt.imshow(img)
plt.show()
