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
		out = sess.run(conv1, {inputs: np.ones((1, 8, 1))})
		print(out)