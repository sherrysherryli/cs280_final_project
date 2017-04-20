# face recognition
import numpy as np
import tensorflow as tf

class batch_norm(object):
  def __init__(self, epsilon=1e-5, momentum = 0.9, name="Batch_norm"):
    with tf.variable_scope(name):
      self.epsilon  = epsilon
      self.momentum = momentum
      self.name = name

  def __call__(self, x, train=True):
    return tf.contrib.layers.batch_norm(x,
                      decay=self.momentum, 
                      updates_collections=None,
                      epsilon=self.epsilon,
                      scale=True,
                      is_training=train,
                      scope=self.name)


class GG(object):
	# Generative neural net
	def __init__(self, sess, input_dim, model_url, batch_size=64, image_size=224, 
		         fully_output_shape=[4,4,128], fully_stddev=0.02, fully_bias_init=0.0,
		         deconv_stride=[2,2], deconv_stddev=0.02):
		'''
		sess: Tensorflow session
		input_dim: Dimension of input vector (number of classes)
		gene_size: Pixels of generated image for x and y
		image_size: Pixels of reshaped image for x and y (after reshaping)
		model_url: Url of a well-trained face recognition neural network.
		'''
		self.sess = sess
		self.input_dim = input_dim
		self.image_size = image_size
		self.model_url = model_url
		self.batch_size = batch_size
		self.fully_output_shape = fully_output_shape
		self.fully_output_size = fully_output_shape[0]*fully_output_shape[1]*fully_output_shape[2]
		self.fully_stddev = fully_stddev
		self.fully_bias_init = fully_bias_init
		self.deconv_stride = deconv_stride
		self.deconv_stddev = deconv_stddev

		self.bnorm0 = batch_norm(name='Bnorm0')
		self.bnorm1 = batch_norm(name='Bnorm1')

		self.init = tf.global_variables_initializer()
		self.load_model()

	def load_model(self):
		with open(self.model_url, mode='rb') as f:
			model = f.read()
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(model)
		image = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, 3])
		tf.import_graph_def(graph_def, input_map={'images': image})
		self.graph_d = tf.get_default_graph()

	def fully(self, u, with_w=False, name='Fully'):
		# u: Input variable of vector
		with tf.variable_scope(name):
			matrix = tf.get_variable('Matrix', [self.input_dim, self.fully_output_size], tf.float32, initializer=tf.random_normal_initializer(stddev=self.fully_stddev))
			bias = tf.get_variable('Bias', [self.fully_output_size], tf.float32, initializer=tf.constant_initializer(self.fully_bias_init))
			f = tf.matmul(u, matrix) + bias
			if with_w:
				return f, matrix, bias
			else:
				return f

	def deconv(self, u, output_shape, i=0, with_w=False, name='Deconv')
		# u: Input variable of 4-D tensor
		ksize = output_shape[0] - (u.get_shape()[0]-1)*self.deconv_stride[i]
		with tf.variable_scope(name):
			kernel = tf.get_variable('Kernel', [ksize, ksize, output_shape[-1], u.get_shape()[-1]], tf.float32, initializer=tf.random_normal_initializer(stddev=self.deconv_stddev))
			d = tf.nn.conv2d_transpose(u, kernel, output_shape, [1, self.deconv_stride[i], self.deconv_stride[i], 1])
			bias = tf.get_variable('Bias', [output_shape[-1]], tf.float32, initializer=tf.constant_initializer(0.0))
			d = tf.reshape(tf.nn.bias_add(d, bias), d.get_shape())
			if with_w:
				return d, kernel, bias
			else:
				return d

	def upsample(self, u):
		# u: Input variable of 4-D tensor
		return tf.image.resize_images(u, [self.image_size, self.image_size], method=ResizeMethod.BICUBIC)

	def generator(self, id):
		with tf.variable_scope('Generator') as scope:
			# fully connected layer f0
			f0 = tf.reshape(self.fully(id, name='Fully0'), [-1]+self.fully_output_shape)
			f0 = tf.nn.relu(self.bnorm0(f0))
			# deconv layer d1
			d1 = self.deconv(f0, [self.batch_size,16,16,64], name='Deconv1')
			d1 = tf.nn.relu(self.bnorm1(d1))
			# deconv layer d2
			d2 = self.deconv(d1, [self.batch_size,32,32,3], name='Deconv2')
			d2 = tf.nn.tanh(d2)
			return self.upsample(d2)

	def discriminator(self, image):
		with tf.Session() as sess:
			prob_tensor = self.graph_d.get_tensor_by_name('import/prob:0')
			prob = sess.run(prob_tensor, feed_dict={image: image})
		return prob

	def build_model(self):
