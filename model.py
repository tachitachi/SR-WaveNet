import tensorflow as tf
import numpy as np

from ops import *

class WaveNet(object):
	def __init__(self, input_size, output_size, dilations, filter_width=2, dilation_channels=32, skip_channels=256, 
		output_channels=256, name='WaveNet', learning_rate=0.001):

		self.input_size = input_size
		self.output_size = output_size
		self.dilations = dilations
		self.filter_width = filter_width
		self.dilation_channels = dilation_channels
		self.skip_channels = skip_channels
		self.output_channels = output_channels
		
		with tf.variable_scope(name):
			self.inputs, self.logits, self.out = self.createNetwork()
			self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

		self.targets = tf.placeholder(tf.float32, [None, self.output_size])

		labels = tf.expand_dims(self.targets, 1)


		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=labels))

		self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

	def createNetwork(self):
		inputs = tf.placeholder(tf.float32, [None, None])
		h = tf.expand_dims(inputs, 2)


		skip_layers = []

		h = DilatedCausalConv1d(h, self.filter_width, channels=self.dilation_channels, dilation_rate=1, name='causal_conv')

		for i in range(len(self.dilations)):
			dilation = self.dilations[i]
			name = 'dilated_conv_{}'.format(i)
			h, skip = ResidualDilationLayer(h, kernel_size=self.filter_width, dilation_channels=self.dilation_channels, 
				skip_channels=self.skip_channels, dilation_rate=dilation, name=name)
			skip_layers.append(skip)


		total = tf.reduce_sum(skip_layers, axis=0)
		total = tf.nn.relu(total)

		total = tf.layers.conv1d(total, filters=self.skip_channels, kernel_size=1, strides=1, padding='SAME')
		total = tf.nn.relu(total)

		total = tf.layers.conv1d(total, filters=self.output_channels, kernel_size=1, strides=1, padding='SAME')

		logits = tf.nn.pool(total, window_shape=(self.input_size,), strides=(1,), pooling_type='AVG', padding='VALID')

		out = tf.nn.softmax(logits)
		
		return inputs, logits, out


	def train(self, inputs, targets):
		sess = tf.get_default_session()
		_, loss = sess.run([self.optimize, self.loss], feed_dict={self.inputs: inputs, self.targets: targets})
		return loss

	def predict(self, inputs):
		sess = tf.get_default_session()
		return sess.run(self.out, feed_dict={self.inputs: inputs})


class SiameseWaveNet():
	def __init__(self, input_size, reduced_dim=2, filter_width=2, num_channels=32, num_layers=10, margin=5.0, name='SiameseWaveNet', learning_rate=0.001):

		self.input_size = input_size
		self.reduced_dim = reduced_dim
		self.filter_width = filter_width
		self.num_channels = num_channels
		self.num_layers = num_layers
		self.margin = margin
		self.name = name

		with tf.variable_scope(name):
			self.inputs_left, self.inputs_right, self.embedding_left, self.embedding_right = self.createNetwork()
			self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)


		self.labels, self.distance, self.loss = self.create_loss()
		self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)



	def createSiamesePart(self, inputs, reuse=False):
		with tf.variable_scope('siamese', reuse=reuse):
			h = tf.expand_dims(inputs, 2)
			for i in range(self.num_layers):
				dilation = 2 ** i
				name = 'conv_{}'.format(i)
				h = DilatedCausalConv1d(h, self.filter_width, channels=self.num_channels, dilation_rate=dilation, name=name)
				h = tf.nn.relu(h)

			# Collapse to 1 channel
			h = tf.reshape(DilatedCausalConv1d(h, 1, channels=1, dilation_rate=1, name='collapse_conv'), [-1, self.input_size])
			h = tf.layers.dense(h, self.reduced_dim)

			return h


	def createNetwork(self, reuse=False):
		inputs_left = tf.placeholder(tf.float32, [None, self.input_size])
		inputs_right = tf.placeholder(tf.float32, [None, self.input_size])

		network_left = self.createSiamesePart(inputs_left, reuse=reuse)
		network_right = self.createSiamesePart(inputs_right, reuse=True)

		return inputs_left, inputs_right, network_left, network_right


	def create_loss(self):

		labels = tf.placeholder(tf.float32, [None])

		# calculate euclidian distance
		# add a small amount to avoid nans
		distance = tf.sqrt(1e-8 + tf.reduce_sum(tf.pow(self.embedding_left - self.embedding_right, 2), axis=1))

		# In the original formula: http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
		# Y = 0 Corresponds to "Same", Y = 1 Corresponds to "different"
		# (1 - Y) * 1/2 * distance^2 + (Y) * 1/2 * (max(0, m - distance))^2
		# In this implementation, y = 1 means "same", and y = 0 means "different"
		# So we flip (1 - Y) and Y
		m = tf.constant(float(self.margin), tf.float32)
		#losses = (1 - labels) * 0.5 * tf.pow(distance, 2) + labels * 0.5 * tf.pow(tf.maximum(0.0, m - distance), 2)
		losses = labels * 0.5 * tf.pow(distance, 2) + (1 - labels) * 0.5 * tf.pow(tf.maximum(0.0, m - distance), 2)

		loss = tf.reduce_mean(losses)

		return labels, distance, loss


	def train(self, inputs_left, inputs_right, labels):
		sess = tf.get_default_session()

		loss, _, distance = sess.run([self.loss, self.optimize, self.distance], feed_dict={
			self.inputs_left: inputs_left,
			self.inputs_right: inputs_right,
			self.labels: labels	
		})

		return loss, distance


	def get_embedding(self, inputs):
		sess = tf.get_default_session()

		embedding = sess.run(self.embedding_left, feed_dict={
			self.inputs_left: inputs
		})

		return embedding

	def get_distance(self, inputs_left, inputs_right):
		sess = tf.get_default_session()

		distance = sess.run(self.distance, feed_dict={
			self.inputs_left: inputs_left,
			self.inputs_right: inputs_right
		})

		return distance
