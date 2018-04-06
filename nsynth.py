import os
import tensorflow as tf
import numpy as np

class NsynthDataReader(object):
	def __init__(self, filepath, batch_size, num_samples=16000, reduced=True, shuffle=True, repeat=True, audio_max_length=64000):


		def _parse_function(example_proto):
			features = {
				"sample_rate": tf.FixedLenFeature([1], dtype=tf.int64),
				#"qualities_str": tf.FixedLenFeature([], dtype=tf.string),
				"note_str": tf.FixedLenFeature([], dtype=tf.string),
				"qualities": tf.FixedLenFeature([10], dtype=tf.int64),
				"audio": tf.FixedLenFeature([audio_max_length], dtype=tf.float32),
				"instrument_family": tf.FixedLenFeature([1], dtype=tf.int64),
				"pitch": tf.FixedLenFeature([1], dtype=tf.int64),
				"instrument_source": tf.FixedLenFeature([1], dtype=tf.int64),
				"instrument_str": tf.FixedLenFeature([], dtype=tf.string),
				"instrument_source_str": tf.FixedLenFeature([], dtype=tf.string),
				"note": tf.FixedLenFeature([1], dtype=tf.int64),
				"instrument": tf.FixedLenFeature([1], dtype=tf.int64),	
				"instrument_family_str": tf.FixedLenFeature([], dtype=tf.string),
				"velocity": tf.FixedLenFeature([1], dtype=tf.int64),
			}
			parsed_features = tf.parse_single_example(example_proto, features)

			if reduced:
				pitch = tf.one_hot(tf.squeeze(parsed_features['pitch']), 128)

				audio = tf.slice(parsed_features['audio'], [0], [num_samples])

				return audio, pitch
			else:
				return parsed_features

		dataset = tf.data.TFRecordDataset(filepath)
		dataset = dataset.map(_parse_function)
		if shuffle:
			dataset = dataset.shuffle(buffer_size=10000)
		if repeat:
			dataset = dataset.repeat()
		dataset = dataset.batch(batch_size)

		self.iterator = dataset.make_initializable_iterator()
		self.iter_n = self.iterator.get_next()

		self.sess = tf.Session()
		self.sess.run(self.iterator.initializer)

	def next(self):
		return self.sess.run(self.iter_n)

