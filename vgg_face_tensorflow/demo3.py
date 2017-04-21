# face recognition
import numpy as np
import tensorflow as tf
import math

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
	def __init__(self, sess, input_dim, model_url, num_id=2622, batch_size=1, image_size=224, 
		         fully_output_shape=[4,4,100], fully_stddev=0.02, fully_bias_init=0.0,
		         deconv_ksize=[5,5], deconv_stride=[2,2], deconv_channel=[50,3], deconv_stddev=0.02,
		         learning_rate=0.001, beta1=0.9, iterates=100):
		'''
		sess: Tensorflow session
		input_dim: Dimension of input vector (number of classes)
		num_id : Total number of identity
		batch_size: Number of image batches
		image_size: Pixels of reshaped image for x and y (after reshaping)
		model_url: Url of a well-trained face recognition neural network.
		'''
		self.sess = sess
		self.input_dim = input_dim
		self.model_url = model_url
		self.num_id = num_id
		self.batch_size = batch_size
		self.image_size = image_size
		self.fully_output_shape = fully_output_shape
		self.fully_output_size = fully_output_shape[0]*fully_output_shape[1]*fully_output_shape[2]
		self.fully_stddev = fully_stddev
		self.fully_bias_init = fully_bias_init
		self.deconv_ksize = deconv_ksize
		self.deconv_stride = deconv_stride
		self.deconv_channel = deconv_channel
		self.deconv_stddev = deconv_stddev
		self.learning_rate = learning_rate
		self.beta1 = beta1
		self.iterates = iterates

		self.bnorm0 = batch_norm(name='Bnorm0')
		self.bnorm1 = batch_norm(name='Bnorm1')

		self.build_model()


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


	def deconv(self, u, output_shape, i=1, with_w=False, name='Deconv'):
		# u: Input variable of 4-D tensor
		with tf.variable_scope(name):
			kernel = tf.get_variable('Kernel', [self.deconv_ksize[i-1], self.deconv_ksize[i-1], output_shape[-1], u.get_shape()[-1]], tf.float32, initializer=tf.random_normal_initializer(stddev=self.deconv_stddev))
			d = tf.nn.conv2d_transpose(u, kernel, output_shape, [1, self.deconv_stride[i-1], self.deconv_stride[i-1], 1])
			bias = tf.get_variable('Bias', [output_shape[-1]], tf.float32, initializer=tf.constant_initializer(0.0))
			d = tf.reshape(tf.nn.bias_add(d, bias), d.get_shape())
			if with_w:
				return d, kernel, bias
			else:
				return d


	def upsample(self, u):
		# u: Input variable of 4-D tensor
		return tf.image.resize_images(u, [self.image_size, self.image_size], method=tf.image.ResizeMethod.BILINEAR)


	def generator(self, id):
		with tf.variable_scope('Generator'):
			# fully connected layer f0
			f0 = tf.reshape(self.fully(id, name='Fully0'), [-1]+self.fully_output_shape)
			f0 = tf.nn.relu(self.bnorm0(f0))
			# deconv layer d1
			a = 1
			output_size1 = self.fully_output_shape[0]*self.deconv_stride[a-1]
			output_shape1 = [self.batch_size, output_size1, output_size1, self.deconv_channel[a-1]]
			d1 = self.deconv(f0, output_shape1, i=1, name='Deconv1')
			d1 = tf.nn.relu(self.bnorm1(d1))
			# deconv layer d2
			a = 2
			output_size2 = output_size1*self.deconv_stride[a-1]
			output_shape2 = [self.batch_size, output_size2, output_size2, self.deconv_channel[a-1]]
			d2 = self.deconv(d1, output_shape2, i=2, name='Deconv2')
			d2 = tf.nn.tanh(d2)
			return d2


	def descriptor(self, image):
		with open(self.model_url, mode='rb') as f:
			model = f.read()
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(model)
		if image.get_shape()[1] == self.image_size:
			tf.import_graph_def(graph_def, input_map={'images': image})
		else:
			tf.import_graph_def(graph_def, input_map={'images': self.upsample(image)})
		return tf.get_default_graph().get_tensor_by_name('import/prob:0')


	def build_model(self):
		self.input_id = tf.placeholder(tf.float32, [self.batch_size, self.input_dim], name='Input_id')
		self.expand_id = tf.placeholder(tf.float32, [self.batch_size, self.num_id], name='Expand_id')
		self.G = self.generator(self.input_id)
		self.D = self.descriptor(self.G)
		self.loss = tf.nn.l2_loss(self.D - self.expand_id)
		var = tf.trainable_variables()
		self.g_var = [v for v in var if 'Generator' in v.name]
		self.saver = tf.train.Saver()


	def train(self):
		# data_x and data_y are labels
		self.data_x = np.identity(self.input_dim, dtype=np.float32)
		self.data_y = np.append(self.data_x, np.zeros((self.input_dim, self.num_id-self.input_dim), dtype=np.float32), axis=1)
		opt = tf.train.AdamOptimizer(self.learning_rate, beta1=self.beta1).minimize(self.loss, var_list=self.g_var)

		for i in xrange(self.iterates):
			j_range = int(math.ceil(self.input_dim/self.batch_size))
			err = 0
			for j in xrange(0, j_range):
				batch_input_id = self.data_x[j*self.batch_size:(j+1)*self.batch_size]
				batch_expand_id = self.data_y[j*self.batch_size:(j+1)*self.batch_size]

				tf.global_variables_initializer().run()
				SEX = self.sess.run(opt, feed_dict={self.input_id: batch_input_id, self.expand_id: batch_expand_id})
				err = err + self.loss.eval({self.input_id: batch_input_id, self.expand_id: batch_expand_id})
			print ('Iterate', i, 'Loss', err/self.input_dim)


with tf.Session() as sess:
	gnn = GG(sess,  
		     model_url='vggface16.tfmodel',
		     input_dim=100, 
		     batch_size=20, 
		     learning_rate=0.001,
		     fully_output_shape=[4,4,128],
		     deconv_stride=[4,4],
		     deconv_channel=[64,3],
		     iterates=1000)
	gnn.train()