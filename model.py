import tensorflow as tf
import numpy as np
import os
import time

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


class WaveNetAutoEncoder(object):
	def __init__(self, input_size, condition_size, output_size, dilations, filter_width=2, encoder_channels=128, dilation_channels=32, skip_channels=256, 
		latent_channels=16, pool_stride=512, name='WaveNetAutoEncoder', learning_rate=0.001):

		self.input_size = input_size
		self.condition_size = condition_size
		self.output_size = output_size
		self.dilations = dilations
		self.filter_width = filter_width
		self.encoder_channels = encoder_channels
		self.dilation_channels = dilation_channels
		self.skip_channels = skip_channels
		self.latent_channels = latent_channels
		self.pool_stride = pool_stride
		
		with tf.variable_scope(name):
			(self.inputs, self.conditions, self.encoding, self.logits, self.out, self.inputs_with_encoding, 
				self.encoding_inputs, self.logits_encoding, self.out_encoding) = self.createNetwork()
			self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

		#self.targets = tf.placeholder(tf.float32, [None, self.output_size])

		self.toFloat = mu_law_decode(tf.argmax(self.out, axis=2), self.output_size)
		self.toFloat_encoding = mu_law_decode(tf.argmax(self.out_encoding, axis=2), self.output_size)
		#self.toFloat_encoding = mu_law_decode(tf.expand_dims(categorical_sample(self.out_encoding[0], self.output_size), 0), self.output_size)

		self.targets = tf.one_hot(mu_law_encode(self.inputs, self.output_size), self.output_size)

		#self.loss = tf.reduce_mean((self.targets - self.out) ** 2)

		#labels = tf.expand_dims(self.targets, 1)


		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.targets))

		self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

		self.saver = tf.train.Saver(self.network_params)
		self.last_checkpoint_time = time.time()


	def createEncoder(self, h, reuse=False):
		with tf.variable_scope('Encoder', reuse=reuse):

			skip_layers_encoder = []
			h, _ = ResidualDilationLayerNC(h, self.filter_width, dilation_channels=self.encoder_channels, skip_channels=self.skip_channels, 
				dilation_rate=1, name='nc_conv')

			for i in range(len(self.dilations)):
				dilation = self.dilations[i]
				name = 'dilated_conv_{}'.format(i)
				h, skip = ResidualDilationLayerNC(h, kernel_size=self.filter_width, dilation_channels=self.encoder_channels, 
					skip_channels=self.skip_channels, dilation_rate=dilation, name=name)
				skip_layers_encoder.append(skip)

			all_skips = tf.reduce_sum(skip_layers_encoder, axis=0)
			reduced_skips = tf.layers.conv1d(all_skips, filters=self.latent_channels, kernel_size=1, strides=1, padding='SAME')

			encoding = tf.nn.pool(reduced_skips, window_shape=(self.pool_stride,), strides=(self.pool_stride,), pooling_type='AVG', padding='VALID')
			return encoding


	def createDecoder(self, h, encoding, conditions, reuse=False):
		with tf.variable_scope('Decoder', reuse=reuse):

			# concatenate the condition to the encoding
			c = tf.expand_dims(conditions, 1)
			c = tf.tile(c, [tf.shape(encoding)[0], tf.shape(encoding)[1], 1])
			encoding_w_condition = tf.concat([encoding, c], axis=2)

			skip_layers_decoder = []


			h = RightShift(h)
			h = DilatedCausalConv1d(h, self.filter_width, channels=self.dilation_channels, dilation_rate=1, name='causal_conv')

			for i in range(len(self.dilations)):
				dilation = self.dilations[i]
				name = 'dilated_conv_{}'.format(i)


				condition_bias = tf.layers.conv1d(encoding_w_condition, filters=self.dilation_channels, kernel_size=1, strides=1, padding='SAME')
				upsampled = ResizeEmbeddingNearestNeighbor(condition_bias, self.pool_stride * tf.shape(condition_bias)[1])

				h = h + upsampled

				h, skip = ResidualDilationLayer(h, kernel_size=self.filter_width, dilation_channels=self.dilation_channels, 
					skip_channels=self.skip_channels, dilation_rate=dilation, name=name)
				skip_layers_decoder.append(skip)


			total = tf.reduce_sum(skip_layers_decoder, axis=0)
			total = tf.nn.relu(total)

			total = tf.layers.conv1d(total, filters=self.skip_channels, kernel_size=1, strides=1, padding='SAME')
			total = tf.nn.relu(total)

			logits = tf.layers.conv1d(total, filters=self.output_size, kernel_size=1, strides=1, padding='SAME')

			#logits = tf.nn.pool(total, window_shape=(self.input_size,), strides=(1,), pooling_type='AVG', padding='VALID')

			out = tf.nn.softmax(logits)

			return logits, out

	def createNetwork(self):
		inputs = tf.placeholder(tf.float32, [None, None])
		conditions = tf.placeholder(tf.float32, [None, self.condition_size])


		inputs_isolated = tf.placeholder(tf.float32, [None, None])
		encoding_isolated = tf.placeholder(tf.float32, [None, None, self.latent_channels])

		h = tf.expand_dims(inputs, 2)
		h2 = tf.expand_dims(inputs_isolated, 2)

		encoding = self.createEncoder(h)

		logits, out = self.createDecoder(h, encoding, conditions)
		logits_from_encoding, out_from_encoding = self.createDecoder(h2, encoding_isolated, conditions, reuse=True)


		
		return inputs, conditions, encoding, logits, out, inputs_isolated, encoding_isolated, logits_from_encoding, out_from_encoding

	def load(self, logdir):
		sess = tf.get_default_session()
		if logdir is not None and os.path.exists(logdir):
			checkpoint_state = tf.train.get_checkpoint_state(logdir)
			if checkpoint_state is not None:
				try:
					self.saver.restore(sess, checkpoint_state.model_checkpoint_path)
					print('Restoring previous session')
					return True
				except (tf.errors.NotFoundError):
					print('Could not find checkpoint at %s', checkpoint_state.model_checkpoint_path)
					return False

	def save(self, logdir, global_step, force=False):
		sess = tf.get_default_session()
		if force or time.time() - self.last_checkpoint_time > 60:
			if not os.path.isdir(logdir):
				os.makedirs(logdir)
			self.saver.save(sess, os.path.join(logdir, 'model.ckpt'), global_step)
			self.last_checkpoint_time = time.time()
			return True

		return False


	def train(self, inputs, conditions):
		sess = tf.get_default_session()
		_, loss = sess.run([self.optimize, self.loss], feed_dict={self.inputs: inputs, self.conditions: conditions})
		return loss

	def encode(self, inputs, conditions):
		sess = tf.get_default_session()
		return sess.run(self.encoding, feed_dict={self.inputs: inputs, self.conditions: conditions})

	def reconstruct(self, inputs, conditions):
		sess = tf.get_default_session()
		return sess.run(self.toFloat, feed_dict={self.inputs: inputs, self.conditions: conditions})

	def reconstruct_with_encoding(self, inputs, conditions, encoding):
		sess = tf.get_default_session()
		return sess.run(self.toFloat_encoding, feed_dict={self.inputs_with_encoding: inputs, self.conditions: conditions,
			self.encoding_inputs: encoding})

	def mu_law(self, inputs, conditions):
		sess = tf.get_default_session()
		return sess.run(self.targets, feed_dict={self.inputs: inputs, self.conditions: conditions})

	def get_logits(self, inputs, conditions, encoding):
		sess = tf.get_default_session()
		return sess.run(self.out_encoding, feed_dict={self.inputs_with_encoding: inputs, self.conditions: conditions,
			self.encoding_inputs: encoding})




class ParallelWaveNet(object):
	def __init__(self, input_size, condition_size, output_size, dilations, teacher, num_flows=2, filter_width=2, dilation_channels=32, skip_channels=256, 
		latent_channels=16, pool_stride=512, name='ParallelWaveNet', learning_rate=0.001):

		self.input_size = input_size
		self.condition_size = condition_size
		self.output_size = output_size
		self.dilations = dilations
		self.teacher = teacher
		self.num_flows = num_flows
		self.filter_width = filter_width
		self.dilation_channels = dilation_channels
		self.skip_channels = skip_channels
		self.latent_channels = latent_channels
		self.pool_stride = pool_stride

		with tf.variable_scope(name):
			self.inputs, self.conditions, self.encoding, self.logits, self.out = self.createNetwork()
			self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

		self.toFloat = mu_law_decode(tf.argmax(self.out, axis=2), self.output_size)

		#log_prob_tf = tf.nn.log_softmax(self.logits)
		#prob_tf = tf.nn.softmax(self.logits)
		self.entropy = tf.reduce_sum(-tf.nn.softmax(self.logits) * tf.nn.log_softmax(self.logits), axis=2)


		#teacher_logits = 

		#self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.teacher.logits_encoding) - self.entropy) 
		self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.teacher.logits_encoding)) 
		self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, var_list=self.network_params)

		self.saver = tf.train.Saver(self.network_params)
		self.last_checkpoint_time = time.time()

	def createPartialFlow(self, inputs, encoding, output_channels, scope):
		with tf.variable_scope(scope):

			# concatenate the condition to the encoding

			skip_layers = []


			h = RightShift(inputs)
			h = DilatedCausalConv1d(h, self.filter_width, channels=self.dilation_channels, dilation_rate=1, name='causal_conv')

			for i in range(len(self.dilations)):
				dilation = self.dilations[i]
				name = 'dilated_conv_{}'.format(i)


				condition_bias = tf.layers.conv1d(encoding, filters=self.dilation_channels, kernel_size=1, strides=1, padding='SAME')
				upsampled = ResizeEmbeddingNearestNeighbor(condition_bias, self.pool_stride * tf.shape(condition_bias)[1])

				# This bias should be added before filter * gate, to each filter and gate
				h = h + upsampled

				h, skip = ResidualDilationLayer(h, kernel_size=self.filter_width, dilation_channels=self.dilation_channels, 
					skip_channels=self.skip_channels, dilation_rate=dilation, name=name)
				skip_layers.append(skip)


			total = tf.reduce_sum(skip_layers, axis=0)
			total = tf.nn.relu(total)

			total = tf.layers.conv1d(total, filters=self.skip_channels, kernel_size=1, strides=1, padding='SAME')
			total = tf.nn.relu(total)

			logits = tf.layers.conv1d(total, filters=output_channels, kernel_size=1, strides=1, padding='SAME')

			return logits


	def createFlow(self, inputs, encoding, output_channels, scope):
		pass

		# take in the current input (1 channel noise)
		# feed through causal dilation layers
		# output 256-way transformation 

		# inputs -> s(z)
		# inputs -> mu(z)

		# output = z * s(z) + mu(z)

		with tf.variable_scope(scope):

			logits_s = self.createPartialFlow(inputs, encoding, output_channels, scope + '_s')
			logits_mu = self.createPartialFlow(inputs, encoding, output_channels, scope + '_mu')

			return inputs * logits_s + logits_mu


	def createNetwork(self):
		#inputs = tf.placeholder(tf.float32, [None, None])
		#conditions = tf.placeholder(tf.float32, [None, self.condition_size])
		#encoding = tf.placeholder(tf.float32, [None, None, self.latent_channels])

		#self.encoding_inputs, self.logits_encoding, self.out_encoding

		inputs = self.teacher.inputs_with_encoding
		conditions = self.teacher.conditions
		encoding = self.teacher.encoding_inputs


		c = tf.expand_dims(conditions, 1)
		c = tf.tile(c, [tf.shape(encoding)[0], tf.shape(encoding)[1], 1])
		encoding_w_condition = tf.concat([encoding, c], axis=2)

		h = tf.expand_dims(inputs, 2)

		for i in range(self.num_flows):
			output_size = self.output_size if i == self.num_flows - 1 else 1
			h = self.createFlow(h, encoding_w_condition, output_size, 'Flow{}'.format(i))

		logits = h
		out = tf.nn.softmax(logits)

		return inputs, conditions, encoding, logits, out

	def load(self, logdir):
		sess = tf.get_default_session()
		if logdir is not None and os.path.exists(logdir):
			checkpoint_state = tf.train.get_checkpoint_state(logdir)
			if checkpoint_state is not None:
				try:
					self.saver.restore(sess, checkpoint_state.model_checkpoint_path)
					print('Restoring previous session')
					return True
				except (tf.errors.NotFoundError):
					print('Could not find checkpoint at %s', checkpoint_state.model_checkpoint_path)
					return False

	def save(self, logdir, global_step, force=False):
		sess = tf.get_default_session()
		if force or time.time() - self.last_checkpoint_time > 60:
			if not os.path.isdir(logdir):
				os.makedirs(logdir)
			self.saver.save(sess, os.path.join(logdir, 'model.ckpt'), global_step)
			self.last_checkpoint_time = time.time()
			return True

		return False


	def generate(self, inputs, conditions, encoding):
		sess = tf.get_default_session()
		return sess.run(self.toFloat, feed_dict={self.inputs: inputs, self.conditions: conditions,
			self.encoding: encoding})

	def getEntropy(self, inputs, conditions, encoding):
		sess = tf.get_default_session()
		return sess.run(self.entropy, feed_dict={self.inputs: inputs, self.conditions: conditions,
			self.encoding: encoding})

	def train(self, inputs, conditions, encoding):
		sess = tf.get_default_session()
		_, loss = sess.run([self.optimize, self.loss], feed_dict={self.inputs: inputs, self.conditions: conditions,
			self.encoding: encoding})

		return loss