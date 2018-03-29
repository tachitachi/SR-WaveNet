import os
import tensorflow as tf
import numpy as np

class NsynthDataReader(object):
	def __init__(self, filepath, batch_size, num_samples=16000):


		def _parse_function(example_proto):
			features = {
				#"note_str": tf.FixedLenFeature([], dtype=tf.string),
				"pitch": tf.FixedLenFeature([1], dtype=tf.int64),
				#"velocity": tf.FixedLenFeature([1], dtype=tf.int64),
				"audio": tf.FixedLenFeature([64000], dtype=tf.float32),
				#"qualities": tf.FixedLenFeature([10], dtype=tf.int64),
				#"instrument_source": tf.FixedLenFeature([1], dtype=tf.int64),
				#"instrument_family": tf.FixedLenFeature([1], dtype=tf.int64),
			}
			parsed_features = tf.parse_single_example(example_proto, features)

			pitch = tf.one_hot(tf.squeeze(parsed_features['pitch']), 128)
			audio = tf.slice(parsed_features['audio'], [0], [num_samples])

			return audio, pitch

		dataset = tf.data.TFRecordDataset(filepath)
		dataset = dataset.map(_parse_function)
		dataset = dataset.shuffle(buffer_size=10000)
		dataset = dataset.repeat()
		dataset = dataset.batch(batch_size)

		self.iterator = dataset.make_initializable_iterator()
		self.iter_n = self.iterator.get_next()

		self.sess = tf.Session()
		self.sess.run(self.iterator.initializer)

	def next(self):
		return self.sess.run(self.iter_n)

