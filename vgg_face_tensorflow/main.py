# face recognition
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


# load mnist data and face reference
mnist_data = input_data.read_data_sets('/tmp/tensorflow/mnist/input_data', one_hot=True)
mnist_ave = np.mean(mnist_data.train.images, axis=0)
mnist_ref = mnist_ave.reshape((1,28,28,1))
face_ave_man = read_reshape_blur_image('ave_face/white_man.jpg', 224, 5)
face_ave_woman = read_reshape_blur_image('ave_face/white_woman.jpg', 224, 5)
face_ave = face_ave_man
face_ref = face_ave.reshape((1,224,224,3))

face_ave_sharpen = sharpen_gray_image(face_ave, 30)


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
	def __init__(self, sess, input_dim, model_url, batch_size=1, learning_target=0, 
		         fully_output_shape=[4,4,256], fully_stddev=0.02, fully_bias_init=0.0,
		         deconv_ksize=[5,5,5,5], deconv_stride=[2,2,2,2], deconv_channel=[128,64,32,3], deconv_stddev=0.02,
		         learning_rate=0.001, epsilon=0.1, iterates=100, ref_image=None, loss_w=1):
		
		# self.sess: Tensorflow session
		# self.input_dim: Dimension of input noise vector 
		# self.num_id : Total number of identity
		# self.batch_size: Number of image batches
		# self.image_size: Pixels of reshaped image for x and y (after reshaping)
		# self.model_url: Url of a well-trained face recognition neural network.
		# self.model0_url: Url of a well-trained discrimation neural network to distinguish face and other images.
		
		self.sess = sess
		self.input_dim = input_dim
		self.model_url = model_url
		self.batch_size = batch_size
		self.learning_target = learning_target
		self.fully_output_shape = fully_output_shape
		self.fully_output_size = fully_output_shape[0]*fully_output_shape[1]*fully_output_shape[2]
		self.fully_stddev = fully_stddev
		self.fully_bias_init = fully_bias_init
		self.deconv_ksize = deconv_ksize
		self.deconv_stride = deconv_stride
		self.deconv_channel = deconv_channel
		self.deconv_stddev = deconv_stddev
		self.learning_rate = learning_rate
		self.epsilon = epsilon
		self.iterates = iterates
		self.ref_image = ref_image
		self.loss_w = loss_w

		self.gene_size = self.fully_output_shape[0]
		for i in xrange(len(self.deconv_stride)):
			self.gene_size = self.gene_size*self.deconv_stride[i] 

		if 'face' in self.model_url:
			self.num_id = 2622
			self.image_size = 224
		else:
			self.num_id = 10
			self.image_size = 28

		self.bnorm0 = batch_norm(name='Bnorm0')
		self.bnorm1 = batch_norm(name='Bnorm1')
		self.bnorm2 = batch_norm(name='Bnorm2')
		self.bnorm3 = batch_norm(name='Bnorm3')

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


	def resample(self, u):
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
			d1 = self.deconv(f0, output_shape1, i=a, name='Deconv1')
			d1 = tf.nn.relu(self.bnorm1(d1))
			# deconv layer d2
			a = 2
			output_size2 = output_size1*self.deconv_stride[a-1]
			output_shape2 = [self.batch_size, output_size2, output_size2, self.deconv_channel[a-1]]
			d2 = self.deconv(d1, output_shape2, i=a, name='Deconv2')
			d2 = tf.nn.relu(self.bnorm2(d2))
			# deconv layer d3
			a = 3
			output_size3 = output_size2*self.deconv_stride[a-1]
			output_shape3 = [self.batch_size, output_size3, output_size3, self.deconv_channel[a-1]]
			d3 = self.deconv(d2, output_shape3, i=a, name='Deconv3')
			d3 = tf.nn.relu(self.bnorm3(d3))
			# deconv layer d4
			a = 4
			output_size4 = output_size3*self.deconv_stride[a-1]
			output_shape4 = [self.batch_size, output_size4, output_size4, self.deconv_channel[a-1]]
			d4 = self.deconv(d3, output_shape4, i=a, name='Deconv4')
			d4 = tf.nn.relu(tf.nn.tanh(d4))

			return self.resample(d4), d4


	def descriptor(self, image):
		with tf.variable_scope('Descriptor') as scope:
			# load 'vggface16.tfmodel'
			if 'face' in self.model_url:
				with open(self.model_url, mode='rb') as f:
					model = f.read()
				graph_def = tf.GraphDef()
				graph_def.ParseFromString(model)
				tf.import_graph_def(graph_def, input_map={'images': image})
				return tf.get_default_graph().get_tensor_by_name('Descriptor/import/prob:0'), tf.get_default_graph().get_tensor_by_name('Descriptor/import/fc8/BiasAdd:0')
			# load 'mnist.tfmodel'
			else:
				with open(self.model_url, mode='rb') as f:
					model = f.read()
				graph_def = tf.GraphDef()
				graph_def.ParseFromString(model)
				tf.import_graph_def(graph_def, input_map={'images': image, 'Deepnn/keep_prob': 1.0})
				return tf.nn.softmax(tf.get_default_graph().get_tensor_by_name('Descriptor/import/Deepnn/prob:0')), tf.get_default_graph().get_tensor_by_name('Descriptor/import/Deepnn/prob:0')


	def build_model(self):
		self.input_noise = tf.placeholder(tf.float32, [self.batch_size, self.input_dim], name='Input_noise')
		self.label = tf.placeholder(tf.float32, [self.batch_size, self.num_id], name='Label')
		self.ref = tf.constant(np.repeat(self.ref_image[0,None], self.batch_size, axis=0), tf.float32)

		self.G, self.G_ = self.generator(self.input_noise)
		self.D, self.D_ = self.descriptor(self.G)
		self.G_sum = tf.summary.image('G_sum', self.G)
		self.D_sum = tf.summary.histogram('D_sum', self.D)

		#self.loss_x = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D, labels=self.label))
		self.loss_x = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D[:,self.learning_target], labels=self.label[:,self.learning_target]))
		#self.loss_l2 = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.G, labels=self.ref))
		self.loss_l2 = tf.nn.l2_loss(self.G - self.ref)/self.batch_size
		self.loss = tf.add(self.loss_x, self.loss_w * self.loss_l2)
		self.loss_x_sum = tf.summary.scalar('Loss_x_sum', self.loss_x)
		self.loss_l2_sum = tf.summary.scalar('Loss_l2_sum', self.loss_l2)
		self.loss_sum = tf.summary.scalar('Loss_sum', self.loss)
		
		var = tf.trainable_variables()
		self.g_var = [v for v in var if 'Generator' in v.name]
		
#		self.saver = tf.train.Saver()


	def train(self):
		# data_label are labels
		data_input_noise = (np.linspace(-1,1,11).reshape(11,1)+np.zeros((11,self.input_dim))).astype(np.float32)
		data_input_noise = np.append(data_input_noise, np.zeros((self.batch_size-11, self.input_dim)), axis=0)
		data_label = np.identity(self.num_id, dtype=np.float32)

		opt = tf.train.AdamOptimizer(self.learning_rate, epsilon=self.epsilon).minimize(self.loss, var_list=self.g_var)
		opt_x = tf.train.AdamOptimizer(self.learning_rate, epsilon=self.epsilon).minimize(self.loss_x, var_list=self.g_var)
		
		tf.global_variables_initializer().run()
		self.sum = tf.summary.merge([self.G_sum,
									 self.D_sum,
									 self.loss_x_sum,
									 self.loss_l2_sum,
									 self.loss_sum])
#		self.writer = tf.summary.FileWriter('./logs', self.sess.graph)
		
		count = 0
		for i in xrange(self.iterates):
			batch_input_noise = np.random.uniform(-1, 1, [self.batch_size,self.input_dim]).astype(np.float32)
			batch_label = np.repeat(data_label[self.learning_target,None], self.batch_size, axis=0)
			if self.loss_l2.eval({self.input_noise: batch_input_noise})>1:
				_, summary_str = self.sess.run([opt, self.sum], 
							                	feed_dict={self.input_noise: batch_input_noise, 
							                               self.label: batch_label})
			else:
				_, summary_str = self.sess.run([opt_x, self.sum], 
							                	feed_dict={self.input_noise: batch_input_noise, 
							                               self.label: batch_label})
#			self.writer.add_summary(summary_str, i+1)
			count += 1
			if count == 10:
				errx = self.loss_x.eval({self.input_noise: batch_input_noise, self.label: batch_label})
				errl2 = self.loss_l2.eval({self.input_noise: batch_input_noise})
				err = self.loss.eval({self.input_noise: batch_input_noise, self.label: batch_label})
				prob = self.D.eval({self.input_noise: batch_input_noise})
				print 'iterates', i+1, 'x-loss:', errx, 'l2-loss', errl2, 'loss', err, 'Prob', np.mean(prob[:,self.learning_target])
				count = 0
		self.pic, self.prob = self.sess.run([self.G, self.D], feed_dict={self.input_noise: data_input_noise})


with tf.Session() as sess:
	'''
	gnn = GG(sess,  
		     model_url='mnist.tfmodel',
		     learning_target=9,
		     input_dim=100, 
		     batch_size=50, 
		     learning_rate=0.01,
		     epsilon=0.1,
		     fully_output_shape=[2,2,256],
		     deconv_stride=[2,2,2,2],
		     deconv_channel=[128,64,32,1],
		     ref_image=mnist_ref, 
		     loss_w=0.005,
		     iterates=1000)
	'''
	gnn = GG(sess,  
		     model_url='vggface16.tfmodel',
		     learning_target=2,
		     input_dim=100, 
		     batch_size=20, 
		     learning_rate=0.01,
		     epsilon=0.1,
		     fully_output_shape=[4,4,256],
		     deconv_stride=[2,2,2,2],
		     deconv_channel=[128,64,32,3],
		     ref_image=face_ref, 
		     loss_w=0.001,
		     iterates=300)

	gnn.train()
	print 'image shape:', gnn.pic.shape
	print 'label shape:', gnn.prob.shape
	print 'learning target:', gnn.learning_target
	print 'first image:', gnn.pic[0].shape, np.min(gnn.pic[0]), np.max(gnn.pic[0])

	if 'face' in gnn.model_url:
		for i in range(11):
			misc.imsave('figures/face_image'+str(i)+'.jpg', gnn.pic[i])
		np.savetxt('figures/number_image.txt', gnn.prob[0:11,gnn.learning_target])
		name = [s.strip() for s in open('names.txt').readlines()]
		for j in range(11):
			rank = np.argsort(gnn.prob[j])[::-1]
			print "image:", str(j)
			print "top 1:", name[rank[0]], gnn.prob[j][rank[0]]
			print "top 2:", name[rank[1]], gnn.prob[j][rank[1]]
			print "top 3:", name[rank[2]], gnn.prob[j][rank[2]]
#			print "top 4:", name[rank[3]], gnn.prob[0][rank[3]]
#			print "top 5:", name[rank[4]], gnn.prob[0][rank[4]]
		f = plt.figure()
		f.add_subplot(2,2,1)
		plt.imshow(gnn.pic[3])
		f.add_subplot(2,2,2)
		plt.imshow(face_ave)
		f.add_subplot(2,2,3)
		plt.imshow(sharpen_gray_image(gnn.pic[3], 30), cmap='gray')
		f.add_subplot(2,2,4)
		plt.imshow(face_ave_sharpen, cmap='gray')
		plt.show()
	else:
		gnn.pic[gnn.pic<0.2]=0
		for i in range(11):
			misc.imsave('figures/Mnist/digit_image'+str(i)+'.jpg', gnn.pic[i,:,:,0])
		np.savetxt('figures/Mnist/prob.txt', gnn.prob[0:11,gnn.learning_target])
		misc.imsave('figures/Mnist/ave.jpg', mnist_ref[0,:,:,0])
		print 'label:', gnn.prob[0:11,gnn.learning_target]
		np.savetxt('number_image.txt', gnn.pic[0])
		f = plt.figure()
		for j in range(11):
			f.add_subplot(3,4,j+1)
			plt.imshow(gnn.pic[j,:,:,0], cmap='gray')
		plt.show()

