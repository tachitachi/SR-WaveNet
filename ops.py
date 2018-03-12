import tensorflow as tf
import numpy as np

# inputs: 3D Tensor -> (batch_size, input_size, channels)
# filters: 3D Tensor -> (kernel_size, input_channels, output_channels)
def _DilatedCausalConv1d(inputs, filters, dilation_rate=1):
	kernel_size = int(filters.shape[0])
	pad_size = dilation_rate * (kernel_size - 1)
	padded = tf.pad(inputs, [[0, 0], [pad_size, 0], [0, 0]])
	return tf.nn.convolution(padded, filters, padding='VALID', dilation_rate=[dilation_rate])


def DilatedCausalConv1d(inputs, kernel_size, channels, dilation_rate=1, name='', dtype=tf.float32, use_bias=True):
	filters = tf.get_variable(name + '_Kernel', [kernel_size, inputs.shape[-1], channels], 
		initializer=tf.contrib.layers.xavier_initializer(), dtype=dtype)
	conv = _DilatedCausalConv1d(inputs, filters, dilation_rate=dilation_rate)
	if use_bias:
		bias = tf.get_variable(name + '_Bias', [1, 1, channels], initializer=tf.constant_initializer(0.0), dtype=dtype)
		conv = conv + bias
	return conv

def ResidualDilationLayer(inputs, kernel_size, dilation_channels, skip_channels, dilation_rate=1, name='', dtype=tf.float32, use_bias=True):

	# input -> causal conv -> tanh
	with tf.variable_scope(name + '_filter'):
		filter_conv = DilatedCausalConv1d(inputs, kernel_size, dilation_channels, dilation_rate, name, dtype, use_bias)
		filter_conv = tf.nn.tanh(filter_conv)

	# input -> causal conv -> sigmoid
	with tf.variable_scope(name + '_gate'):
		gated_conv = DilatedCausalConv1d(inputs, kernel_size, dilation_channels, dilation_rate, name, dtype, use_bias)
		gated_conv = tf.nn.sigmoid(filter_conv)


	combined = filter_conv * gated_conv

	# 1x1 convolution
	residual = tf.layers.conv1d(combined, filters=dilation_channels, kernel_size=1, strides=1, padding='SAME')
	dense = inputs + residual


	# 1x1 convolution
	skip = tf.layers.conv1d(combined, filters=skip_channels, kernel_size=1, strides=1, padding='SAME')

	return dense, skip

def ResidualDilationLayerNC(inputs, kernel_size, dilation_channels, skip_channels, dilation_rate=1, name='', dtype=tf.float32, use_bias=True):
	x = tf.nn.relu(inputs)
	with tf.variable_scope(name + '_NC'):
		x = tf.layers.conv1d(x, filters=dilation_channels, kernel_size=kernel_size, strides=1, padding='SAME')
		x = tf.nn.relu(x)

	residual = tf.layers.conv1d(x, filters=dilation_channels, kernel_size=1, strides=1, padding='SAME')
	skip = tf.layers.conv1d(x, filters=skip_channels, kernel_size=1, strides=1, padding='SAME')

	return residual, skip





# [batch_size, embedding_size, embedding_channels]
def ResizeEmbeddingNearestNeighbor(inputs, output_size):
	embedding_size = inputs.shape[1]
	embedding_channels = inputs.shape[2]

	# Add fake channel size of 1
	reshaped = tf.expand_dims(inputs, 3)


	resized = tf.image.resize_nearest_neighbor(reshaped, [output_size, embedding_channels])

	return tf.squeeze(resized, axis=[3])


# Right shift a 3D tensor
def RightShift(inputs, shift_size=1):
	p = tf.pad(inputs, [[0, 0], [shift_size, 0], [0, 0]])
	return tf.slice(p, [0, 0, 0], [-1, tf.shape(p)[1]-shift_size, inputs.shape[2]])

def mu_law_encode(audio, quantization_channels):
	'''Quantizes waveform amplitudes.'''
	with tf.name_scope('encode'):
		mu = tf.to_float(quantization_channels - 1)
		# Perform mu-law companding transformation (ITU-T, 1988).
		# Minimum operation is here to deal with rare large amplitudes caused
		# by resampling.
		safe_audio_abs = tf.minimum(tf.abs(audio), 1.0)
		magnitude = tf.log1p(mu * safe_audio_abs) / tf.log1p(mu)
		signal = tf.sign(audio) * magnitude
		# Quantize signal to the specified number of levels.
		return tf.to_int32((signal + 1) / 2 * mu + 0.5)


def mu_law_decode(output, quantization_channels):
	'''Recovers waveform from quantized values.'''
	with tf.name_scope('decode'):
		mu = quantization_channels - 1
		# Map values back to [-1, 1].
		signal = 2 * (tf.to_float(output) / mu) - 1
		# Perform inverse of mu-law transformation.
		magnitude = (1 / mu) * ((1 + mu)**abs(signal) - 1)
		return tf.sign(signal) * magnitude

def _flatten(x):
	return np.reshape(x, [-1, np.prod(list(x.shape)[1:])])


if __name__ == '__main__':
	# Unit testing for ops

	x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
	f1 = np.array([1, 1], dtype=np.float32)
	f2 = np.array([1, 0, 1], dtype=np.float32)
	f3 = np.array([1, 0, 0, 0, 1], dtype=np.float32)

	f4 = np.array([[1, 2, 1, 2]], dtype=np.float32)

	inputs = tf.placeholder(tf.float32, [None, 8, 1])
	conv1 = DilatedCausalConv1d(inputs, kernel_size=3, channels=4, dilation_rate=4, name='causal_conv1')

	dense, skip = ResidualDilationLayer(inputs, kernel_size=2, dilation_channels=8, skip_channels=4, dilation_rate=4, name='dilation_layer1')

	total = dense + skip

	print(tf.get_collection(tf.GraphKeys.VARIABLES))

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		print(_flatten(sess.run(_DilatedCausalConv1d(np.reshape(x, [1, -1, 1]), np.reshape(f1, [2, 1, 1])))))
		print(_flatten(sess.run(_DilatedCausalConv1d(np.reshape(x, [1, -1, 1]), np.reshape(f2, [3, 1, 1])))))
		print(_flatten(sess.run(_DilatedCausalConv1d(np.reshape(x, [1, -1, 1]), np.reshape(f3, [5, 1, 1])))))
		print(_flatten(sess.run(_DilatedCausalConv1d(np.reshape(x, [1, -1, 1]), np.reshape(f1, [2, 1, 1]), dilation_rate=2))))
		print(_flatten(sess.run(_DilatedCausalConv1d(np.reshape(x, [1, -1, 1]), np.reshape(f1, [2, 1, 1]), dilation_rate=3))))
		print(_flatten(sess.run(_DilatedCausalConv1d(np.reshape(x, [1, -1, 1]), np.reshape(f1, [2, 1, 1]), dilation_rate=4))))
		print(_flatten(sess.run(_DilatedCausalConv1d(np.reshape(x, [1, -1, 1]), np.reshape(f1, [2, 1, 1]), dilation_rate=6))))


		print(sess.run(_DilatedCausalConv1d(np.reshape(x, [1, -1, 1]), np.reshape(f4, [2, 1, 2]), dilation_rate=1)))

		print(sess.run(tf.nn.convolution(np.reshape(x, [1, -1, 1]), np.reshape(f4, [2, 1, 2]), padding='VALID', dilation_rate=[1])))


		out = sess.run(conv1, {inputs: np.random.random((1, 8, 1))})
		print(out)
		out = sess.run(dense, {inputs: np.ones((1, 8, 1))})
		print(out)
		out = sess.run(skip, {inputs: np.ones((1, 8, 1))})
		print(out)
		out = sess.run(total, {inputs: np.ones((1, 8, 1))})
		print(out)