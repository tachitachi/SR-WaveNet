import numpy as np
from scipy.io import wavfile
import tensorflow as tf
import os
import json

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

if __name__ == '__main__':

	tfrecord_filename = 'synthetic_valid.tfrecord'

	rootdir = os.path.join('nsynth_data', 'nsynth-valid')

	writer = tf.python_io.TFRecordWriter(os.path.join(rootdir, tfrecord_filename))


	with open(os.path.join(rootdir, 'examples.json')) as json_data:
		d = json.load(json_data)
		json_data.close()

		count = 0
		
		for key in d:
			data = d[key]

			instrument_filename = data['note_str']
			pitch = data['pitch']
			instrument_source = data['instrument_source']

			if instrument_source != 2:
				continue

			#print(data)

			_, audio_data = wavfile.read(os.path.join(rootdir, 'audio', instrument_filename + '.wav'))

			audio_data = audio_data / 32767.0

			feature = {'pitch': _int64_feature(pitch),
				'audio': _float_feature(audio_data)}

			example = tf.train.Example(features=tf.train.Features(feature=feature))

			writer.write(example.SerializeToString())

			if count % 100 == 0:
				print(count)

			count += 1


	writer.close()
