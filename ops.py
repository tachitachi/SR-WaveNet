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

# TODO: How to properly add bias from conditioning?
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
	dense = (inputs + residual) * 0.7071067811865476


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

def categorical_sample(logits, d):
    value = tf.squeeze(tf.multinomial(logits - tf.reduce_max(logits, [1], keepdims=True), 1), [1])
    return value
    #return tf.one_hot(value, d)

def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis, keepdims=True)
    return x - m - tf.log(tf.reduce_sum(tf.exp(x-m), axis, keepdims=True))

def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    axis = len(x.get_shape())-1
    m = tf.reduce_max(x, axis)
    m2 = tf.reduce_max(x, axis, keepdims=True)
    return m + tf.log(tf.reduce_sum(tf.exp(x-m2), axis))

def discretized_mix_logistic_loss(x,l,sum_all=True):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    #xs = int_shape(x) # true image (i.e. labels) to regress to, e.g. (B,32,32,3)
    #xs = [-1, int(l.shape[1]), 1] # audio shape [B, L, 1]
    #xs_t = tf.shape(x)
    #ls = int_shape(l) # predicted distribution, e.g. (B,32,100)
    #nr_mix = int(ls[-1] / 10) 
    nr_mix = int(int(l.shape[-1]) / 4) # here and below: unpacking the params of the mixture of logistics
    logit_probs = l[:,:,:nr_mix]
    #l = tf.reshape(l[:,:,nr_mix:], xs + [nr_mix*3])
    l = tf.expand_dims(l[:, :, nr_mix:], 2)
    means = l[:,:,:,:nr_mix]
    log_scales = tf.maximum(l[:,:,:,nr_mix:2*nr_mix], -7.)
    coeffs = tf.nn.tanh(l[:,:,:,2*nr_mix:3*nr_mix])
    #x = tf.reshape(x, xs + [1]) + tf.zeros([xs_t[:2]] + [nr_mix]) # here and below: getting the means and adjusting them based on preceding sub-pixels
    x = tf.tile(tf.expand_dims(x, 3), [1, 1, 1, nr_mix])


    #m2 = tf.reshape(means[:,:,1,:] + coeffs[:, :, 0, :] * x[:, :, 0, :], [xs_t[0],xs[1],1,nr_mix])
    #m3 = tf.reshape(means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * x[:, :, :, 0, :] + coeffs[:, :, :, 2, :] * x[:, :, :, 1, :], [xs[0],xs[1],xs[2],1,nr_mix])
    #means = tf.concat([tf.reshape(means[:,:,:,0,:], [xs[0],xs[1],xs[2],1,nr_mix]), m2, m3],3)
    #means = tf.reshape(means[:,:,0,:], [xs[0],xs[1],1,nr_mix])
    centered_x = x - means
    inv_stdv = tf.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1./255.)
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1./255.)
    cdf_min = tf.nn.sigmoid(min_in)
    log_cdf_plus = plus_in - tf.nn.softplus(plus_in) # log probability for edge case of 0 (before scaling)
    log_one_minus_cdf_min = -tf.nn.softplus(min_in) # log probability for edge case of 255 (before scaling)
    cdf_delta = cdf_plus - cdf_min # probability for all other cases
    mid_in = inv_stdv * centered_x
    log_pdf_mid = mid_in - log_scales - 2.*tf.nn.softplus(mid_in) # log probability in the center of the bin, to be used in extreme cases (not actually used in our code)

    # now select the right output: left edge case, right edge case, normal case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation based on the assumption that the log-density is constant in the bin of the observed sub-pixel value
    log_probs = tf.where(x < -0.999, log_cdf_plus, tf.where(x > 0.999, log_one_minus_cdf_min, tf.where(cdf_delta > 1e-5, tf.log(tf.maximum(cdf_delta, 1e-12)), log_pdf_mid - np.log(127.5))))

    log_probs = tf.reduce_sum(log_probs,2) + log_prob_from_logits(logit_probs)

    if sum_all:
        return -tf.reduce_sum(log_sum_exp(log_probs))
    else:
        #return -tf.reduce_sum(log_sum_exp(log_probs),1)
        return -tf.expand_dims(log_sum_exp(log_probs), 2)


def sample_from_discretized_mix_logistic(l,nr_mix):
    #ls = int_shape(l)
    #xs = [-1, int(l.shape[1]), 1] # audio shape [B, L, 1]
    #xs_t = tf.shape(l)
    # unpack parameters
    logit_probs = l[:, :, :nr_mix]
    #l = tf.reshape(l[:, :, nr_mix:], xs_t[:] + [nr_mix*3])
    l = tf.expand_dims(l[:, :, nr_mix:], 2)
    # sample mixture indicator from softmax
    sel = tf.one_hot(tf.argmax(logit_probs - tf.log(-tf.log(tf.random_uniform(tf.shape(logit_probs), minval=1e-5, maxval=1. - 1e-5))), 2), depth=nr_mix, dtype=tf.float32)
    #sel = tf.reshape(sel, xs_t[:-1] + [1,nr_mix])
    sel = tf.expand_dims(sel, 2)
    # select logistic parameters
    means = tf.reduce_sum(l[:,:,:,:nr_mix]*sel,3)
    log_scales = tf.maximum(tf.reduce_sum(l[:,:,:,nr_mix:2*nr_mix]*sel,3), -7.)
    coeffs = tf.reduce_sum(tf.nn.tanh(l[:,:,:,2*nr_mix:3*nr_mix])*sel,3)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = tf.random_uniform(tf.shape(means), minval=1e-5, maxval=1. - 1e-5)
    x = means + tf.exp(log_scales)*(tf.log(u) - tf.log(1. - u))
    #x0 = tf.minimum(tf.maximum(x[:,:,0], -1.), 1.)
    x = tf.minimum(tf.maximum(x, -1.), 1.)
    #return x0
    return x

def probs_logistic(scale, mu, y, num_classes=256, log_scale_min=-14):
    means = mu
    scale = tf.clip_by_value(scale,np.exp(log_scale_min),np.inf)
    centered_y = y - means
    inv_stdv = 1/scale
    plus_in = inv_stdv * (centered_y + 1. / (num_classes - 1))
    cdf_plus = tf.nn.sigmoid(plus_in)
    min_in = inv_stdv * (centered_y - 1. / (num_classes - 1))
    cdf_min = tf.nn.sigmoid(min_in)
    cdf_delta = cdf_plus - cdf_min
    return cdf_delta
    #return tf.clip_by_value(cdf_delta,1e-5, 1.)


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