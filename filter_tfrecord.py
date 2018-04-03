import numpy as np
import tensorflow as tf
import os
from nsynth import NsynthDataReader

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

if __name__ == '__main__':

	data = NsynthDataReader('nsynth_data/nsynth-train.tfrecord', 1, num_samples=64000, reduced=False, shuffle=False, repeat=False)

	writer = tf.python_io.TFRecordWriter(os.path.join('nsynth_data', 'filtered_note60.tfrecord'))

	count = 0
	try:
		while True:
			if count % 1000 == 0:
				print(count)
			count += 1

			d = data.next()

			if d['pitch'][0][0] != 60:
				continue

			feature = {
				'sample_rate': _int64_feature(d['sample_rate'][0]),
				'note_str': _bytes_feature(d['note_str'][0]),
				'qualities': _int64_feature(d['qualities'][0]),
				'audio': _float_feature(d['audio'][0]),
				'instrument_family': _int64_feature(d['instrument_family'][0]),
				'pitch': _int64_feature(d['pitch'][0]),
				'instrument_source': _int64_feature(d['instrument_source'][0]),
				'instrument_str': _bytes_feature(d['instrument_str'][0]),
				'instrument_source_str': _bytes_feature(d['instrument_source_str'][0]),
				'note': _int64_feature(d['note'][0]),
				'instrument': _int64_feature(d['instrument'][0]),
				'instrument_family_str': _bytes_feature(d['instrument_family_str'][0]),
				'velocity': _int64_feature(d['velocity'][0]),
			}

			example = tf.train.Example(features=tf.train.Features(feature=feature))

			writer.write(example.SerializeToString())


	except tf.errors.OutOfRangeError as e:
		print('Finished writing {} items'.format(count))


	writer.close()
