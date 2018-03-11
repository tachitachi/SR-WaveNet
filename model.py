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

