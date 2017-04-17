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


pix = 224
# read image	
img = reshape_image('zhi.jpg', pix)
# load model
with open('vggface16.tfmodel', mode='rb') as f:
	fileContent = f.read()


graph_def = tf.GraphDef()
graph_def.ParseFromString(fileContent)
images = tf.placeholder('float', [None, pix, pix, 3])
tf.import_graph_def(graph_def, input_map={ 'images': images })
print "graph loaded from disk"
graph = tf.get_default_graph()


init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
  	print "variables initialized"

  	batch = img.reshape((1, pix, pix, 3))
  	assert batch.shape == (1, pix, pix, 3)

  	feed_dict = { images: batch }

  	prob_tensor = graph.get_tensor_by_name('import/prob:0')
  	prob = sess.run(prob_tensor, feed_dict=feed_dict)


print "prob shape", prob.shape
name = [s.strip() for s in open('names.txt').readlines()]
rank = np.argsort(prob[0])[::-1]
print "top 1:", name[rank[0]], prob[0][rank[0]]
print "top 2:", name[rank[1]], prob[0][rank[1]]
print "top 3:", name[rank[2]], prob[0][rank[2]]
print "top 4:", name[rank[3]], prob[0][rank[3]]
print "top 5:", name[rank[4]], prob[0][rank[4]]
plt.figure()
plt.imshow(img)
plt.show()
