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
	def __init__(self, input_size, condition_size, num_mixtures, dilations, filter_width=2, encoder_channels=128, dilation_channels=32, skip_channels=256, 
		latent_channels=16, pool_stride=512, name='WaveNetAutoEncoder', learning_rate=0.001):

		self.input_size = input_size
		self.condition_size = condition_size
		self.num_mixtures = num_mixtures
		self.dilations = dilations
		self.filter_width = filter_width
		self.encoder_channels = encoder_channels
		self.dilation_channels = dilation_channels
		self.skip_channels = skip_channels
		self.latent_channels = latent_channels
		self.pool_stride = pool_stride


		self.graph = tf.Graph()
		with self.graph.as_default():
			
			with tf.variable_scope(name):
				self.createNetwork()
				self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)

			#self.targets = tf.placeholder(tf.float32, [None, self.output_size])

			#self.toFloat = mu_law_decode(tf.argmax(self.out, axis=2), self.output_size)
			#self.toFloat_encoding = mu_law_decode(tf.argmax(self.out_from_encoding, axis=2), self.output_size)
			#self.toFloat_encoding = mu_law_decode(tf.expand_dims(categorical_sample(self.out_from_encoding[0], self.output_size), 0), self.output_size)

			#self.targets = tf.one_hot(mu_law_encode(self.inputs, self.output_size), self.output_size)

			#self.loss = tf.reduce_mean((self.targets - self.out) ** 2)

			labels = tf.expand_dims(self.inputs, 2)
			labels_truth = tf.expand_dims(self.inputs_truth, 2)


			#self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.logits, labels=self.targets))

			self.loss = discretized_mix_logistic_loss(labels, self.logits)
			self.loss_encoding = discretized_mix_logistic_loss(labels_truth, self.logits_from_encoding)

			self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

			self.saver = tf.train.Saver(self.network_params)
			self.last_checkpoint_time = time.time()

			tf.add_to_collection('Inputs_e', self.inputs)
			tf.add_to_collection('Conditions', self.conditions)
			tf.add_to_collection('Encoding_output', self.encoding)
			tf.add_to_collection('Logits_e', self.logits)
			tf.add_to_collection('Out_e', self.out)
			tf.add_to_collection('Loss_e', self.loss)

			tf.add_to_collection('Encoding_input', self.encoding_isolated)
			tf.add_to_collection('Logits_d', self.logits_from_encoding)
			tf.add_to_collection('Out_d', self.out_from_encoding)
			#tf.add_to_collection('Loss_d', self.loss_encoding)

			tf.add_to_collection('Inputs_truth', self.inputs_truth)


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


	def createDecoder(self, truth, encoding, conditions, reuse=False):
		with tf.variable_scope('Decoder', reuse=reuse):

			# concatenate the condition to the encoding
			c = tf.expand_dims(conditions, 1)
			c = tf.tile(c, [tf.shape(encoding)[0], tf.shape(encoding)[1], 1])
			encoding_w_condition = tf.concat([encoding, c], axis=2)

			skip_layers_decoder = []


			h = RightShift(truth)
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

			logits = tf.layers.conv1d(total, filters=self.num_mixtures * 4, kernel_size=1, strides=1, padding='SAME')

			out = tf.squeeze(sample_from_discretized_mix_logistic(logits, self.num_mixtures), axis=2)

			return logits, out

	def createNetwork(self):
		self.inputs = tf.placeholder(tf.float32, [None, None], 'inputs_placeholder')
		self.inputs_truth = tf.placeholder(tf.float32, [None, None], 'inputs_truth_placeholder')
		self.conditions = tf.placeholder(tf.float32, [None, self.condition_size], 'conditions_placeholder')

		self.encoding_isolated = tf.placeholder(tf.float32, [None, None, self.latent_channels], 'encoding_nodecoder_placeholder')

		h = tf.expand_dims(self.inputs, 2)
		h_truth = tf.expand_dims(self.inputs_truth, 2)

		self.encoding = self.createEncoder(h)

		self.logits, self.out = self.createDecoder(h_truth, self.encoding, self.conditions)
		self.logits_from_encoding, self.out_from_encoding = self.createDecoder(h_truth, self.encoding_isolated, self.conditions, reuse=True)

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
		_, loss = sess.run([self.optimize, self.loss], feed_dict={self.inputs: inputs, self.inputs_truth: inputs, self.conditions: conditions})
		return loss

	def encode(self, inputs, conditions):
		sess = tf.get_default_session()
		return sess.run(self.encoding, feed_dict={self.inputs: inputs, self.conditions: conditions})

	def reconstruct(self, inputs, conditions):
		sess = tf.get_default_session()
		return sess.run(self.out, feed_dict={self.inputs: inputs, self.c: inputs, self.conditions: conditions})

	def reconstruct_with_encoding(self, inputs, conditions, encoding):
		sess = tf.get_default_session()
		return sess.run(self.out_from_encoding, feed_dict={self.inputs_truth: inputs, self.conditions: conditions,
			self.encoding_isolated: encoding})

	def mu_law(self, inputs, conditions):
		sess = tf.get_default_session()
		return sess.run(self.targets, feed_dict={self.inputs: inputs, self.conditions: conditions})

	def get_logits(self, inputs, conditions, encoding):
		sess = tf.get_default_session()
		return sess.run(self.logits_from_encoding, feed_dict={self.inputs_truth: inputs, self.conditions: conditions,
			self.encoding_isolated: encoding})




class ParallelWaveNet(object):
	def __init__(self, input_size, condition_size, dilations, teacher, num_flows=2, filter_width=2, dilation_channels=32, skip_channels=256, 
		latent_channels=16, pool_stride=512, name='ParallelWaveNet', learning_rate=0.001):

		self.input_size = input_size
		self.condition_size = condition_size
		#self.num_mixtures = num_mixtures
		self.dilations = dilations
		
		self.teacher = teacher # directory for a teacher to load
		
		self.num_flows = num_flows
		self.filter_width = filter_width
		self.dilation_channels = dilation_channels
		self.skip_channels = skip_channels
		self.latent_channels = latent_channels
		self.pool_stride = pool_stride

		self.graph = tf.Graph()
		with self.graph.as_default():

			with tf.variable_scope(name):
				self.createNetwork()
				self.network_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, tf.get_variable_scope().name)


			# load teacher network to get loss

			#logdir = 'teachers/1521322640552'
			checkpoint_state = tf.train.get_checkpoint_state(self.teacher)
			#teacher_meta = tf.train.import_meta_graph(checkpoint_state.model_checkpoint_path + '.meta')

			# replace the front of the teacher network, to connect the end of the student network
			self.teacher_meta = tf.train.import_meta_graph(checkpoint_state.model_checkpoint_path + '.meta', 
				input_map={
					'WaveNetAutoEncoder/inputs_nodecoder_placeholder:0': self.inputs_teacher,
					'WaveNetAutoEncoder/conditions_placeholder:0': self.conditions,
					'WaveNetAutoEncoder/encoding_nodecoder_placeholder:0': self.encoding
					})

			# Prevent gradients from flowing through teacher network?
			teacher_logits =  tf.stop_gradient(self.graph.get_collection('Logits_d')[0])
			self.teacher_encoding = self.graph.get_collection('Encoding_output')[0]

			self.teacher_inputs = self.graph.get_collection('Inputs_e')[0]
			#self.conditions = graph.get_collection('Conditions')[0]
			self.teacher_out = self.graph.get_collection('Out_e')[0]
			#logits =  graph.get_collection('Logits_e')[0]

			#log_prob_tf = tf.nn.log_softmax(self.logits)
			#prob_tf = tf.nn.softmax(self.logits)

			#self.entropy = tf.reduce_sum(-tf.nn.softmax(self.logits) * tf.nn.log_softmax(self.logits), axis=2)

			# entropy
			# E ( sum ln s(z, theta) ) + 2T

			# Teacher uses a mixture of logistics, but student only has one logistic

			# mixture params: [B, param (2), mixture_num]  param: [scale, mean]

			self.entropy = tf.reduce_sum(tf.log(self.s_tot)) + tf.cast(2 * tf.shape(self.inputs)[0], tf.float32)

			#print(self.out)

			# the output of the student network should flow through the teacher network
			self.loss = discretized_mix_logistic_loss(teacher_logits, self.out)

			self.optimize = tf.train.AdamOptimizer(learning_rate).minimize(self.loss, var_list=self.network_params)

			self.saver = tf.train.Saver(self.network_params)
			self.last_checkpoint_time = time.time()

	def createPartialFlow(self, inputs, encoding, scope):
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


#			total = tf.reduce_sum(skip_layers, axis=0)
#			total = tf.nn.relu(total)
#
#			total = tf.layers.conv1d(total, filters=self.skip_channels, kernel_size=1, strides=1, padding='SAME')
#			total = tf.nn.relu(total)
#
#			logits = tf.layers.conv1d(total, filters=2, kernel_size=1, strides=1, padding='SAME')

			h = tf.nn.relu(h)
			h = tf.layers.conv1d(h, filters=2, kernel_size=1, strides=1, padding='SAME')

			return h


	def createFlow(self, inputs, encoding, scope):
		pass

		# take in the current input (1 channel noise)
		# feed through causal dilation layers
		# output 256-way transformation 

		# inputs -> s(z)
		# inputs -> mu(z)

		# output = z * s(z) + mu(z)

		with tf.variable_scope(scope):

			#logits_s = self.createPartialFlow(inputs, encoding, output_channels, scope + '_s')
			#logits_mu = self.createPartialFlow(inputs, encoding, output_channels, scope + '_mu')

			#return inputs * logits_s + logits_mu

			params = self.createPartialFlow(inputs, encoding, scope)
			#out = sample_from_discretized_mix_logistic(logits, self.num_mixtures)

			scale = tf.exp(tf.slice(params, [0, 0, 0], [-1, -1, 1]))
			mean = tf.slice(params, [0, 0, 1], [-1, -1, 1])

			#out = inputs * scale + mean

			#return params#, out

			return scale, mean


	def createNetwork(self):
		# input logistic noise
		self.inputs = tf.placeholder(tf.float32, [None, None])
		self.conditions = tf.placeholder(tf.float32, [None, self.condition_size])
		self.encoding = tf.placeholder(tf.float32, [None, None, self.latent_channels])

		c = tf.expand_dims(self.conditions, 1)
		c = tf.tile(c, [tf.shape(self.encoding)[0], tf.shape(self.encoding)[1], 1])
		encoding_w_condition = tf.concat([self.encoding, c], axis=2)

		x = tf.expand_dims(self.inputs, 2)

		#self.param_list = []

		scales = []
		means = []

		for i in range(self.num_flows):
			scale, mean = self.createFlow(x, encoding_w_condition, 'Flow{}'.format(i))
			#self.param_list.append(params)
			scales.append(scale)
			means.append(mean)

		#self.s_tot = self.param_list[0][:,:,0]

		self.s_tot = tf.ones([tf.shape(self.inputs)[0], tf.shape(self.inputs)[0], 1])
		self.mu_tot = tf.zeros([tf.shape(self.inputs)[0], tf.shape(self.inputs)[0], 1])

		for i in range(len(scales)):
			self.s_tot *= scales[i]

			print('multiplying s_{}'.format(i), scales[i])

			mu = means[i]

			for j in range(i + 1, len(scales)):
				print('multiplying mu_{} * s_{}'.format(i, j))
				mu *= scales[j]

			print('adding mu_{}'.format(i), means[i])

			self.mu_tot += mu

		#self.out = probs_logistic(self.s_tot, self.mu_tot, x)
		self.out = x * self.s_tot + self.mu_tot

		print('@@@@ OUT', self.s_tot, self.mu_tot, x, self.out)


	def load(self, logdir):
		sess = tf.get_default_session()

		teacher_checkpoint_state = tf.train.get_checkpoint_state(self.teacher)
		self.teacher_meta.restore(sess, teacher_checkpoint_state.model_checkpoint_path)

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
		return sess.run(self.out, feed_dict={self.inputs: inputs, self.conditions: conditions,
			self.encoding: encoding})

	def getEntropy(self, inputs, conditions, encoding):
		sess = tf.get_default_session()
		return sess.run([self.entropy], feed_dict={self.inputs: inputs, self.conditions: conditions,
			self.encoding: encoding})

	def train(self, inputs, conditions, encoding):
		sess = tf.get_default_session()
		_, loss = sess.run([self.optimize, self.loss], feed_dict={self.inputs: inputs, self.conditions: conditions,
			self.encoding: encoding})

		return loss

	def encode(self, inputs, conditions):
		sess = tf.get_default_session()
		return sess.run(self.teacher_encoding, feed_dict={self.teacher_inputs: inputs, self.conditions: conditions})

	def reconstruct(self, inputs, conditions):
		sess = tf.get_default_session()
		return sess.run(self.teacher_out, feed_dict={self.teacher_inputs: inputs, self.conditions: conditions})