import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data import AudioData
from model import WaveNet


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--classifier', action='store_true', help='Train a classifier wavenet')
	parser.add_argument('--siamese', action='store_true', help='Train a siamese wavenet')
	args = parser.parse_args()

	batch_size = 1
	num_steps = 100000
	print_steps = 100

	if args.classifier:

		audio_data = AudioData()
		num_samples = audio_data.num_samples
		num_classes = audio_data.classes

		network = WaveNet(num_samples, num_classes, output_channels=num_classes)

		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())

			for i in range(num_steps):
				x, y = audio_data.TrainBatch(batch_size)
				loss = network.train(x, y)

				if i % print_steps == 0:
					labels = network.predict(x)
					#print(loss, np.sum(np.abs(y - np.round(labels))))
					print(loss)
					print(y - labels)


	if args.siamese:

		pass