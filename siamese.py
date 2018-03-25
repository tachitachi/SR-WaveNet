import argparse
import time
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import SiameseWaveNet
from simple_audio import generate_random_wave

from nsynth import NsynthDataReader



if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--logdir', type=str, default='siamese/%d' % int(time.time() * 1000), help='Directory where checkpoint and summary is stored')
	parser.add_argument('--start', type=int, default=0, help='Starting index')
	parser.add_argument('--train', action='store_true', help='Train siamese network')
	parser.add_argument('--test', action='store_true', help='Test siamese network')

	args = parser.parse_args()

	batch_size = 1
	num_steps = 1000000
	print_steps = 100

	last_checkpoint_time = time.time()

	#audio_data = AudioData()
	#audio_data = NsynthDataReader(os.path.join('nsynth_data', 'nsynth-train.tfrecord'), batch_size)
	#num_samples = 64000
	#num_classes = 128

	num_samples = 5120

	dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
              1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
              1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    # input_size, output_dimensions, dilations, margin=5.0, filter_width=2, dilation_channels=32, 
	#	skip_channels=256, name='SiameseWaveNet', learning_rate=0.001
	network = SiameseWaveNet(input_size=num_samples, output_dimensions=2, dilations=dilations, 
		skip_channels=128, learning_rate=1e-4)

	with tf.Session(graph=network.graph) as sess:
		sess.run(tf.global_variables_initializer())

		network.load(sess, args.logdir)


		if args.train:
			for global_step in range(args.start, num_steps):

				x1, y1 = generate_random_wave(num_samples)
				x2, y2 = generate_random_wave(num_samples)

				labels = (y1 == y2).all().astype(np.float32)

				loss, distance = network.train(sess, [x1], [x2], [labels])

				if global_step % print_steps == 0:
					print(global_step, loss, distance, labels)


				# Checkpoint once per minute
				network.save(sess, args.logdir, global_step, force=False)

			network.save(sess, args.logdir, global_step, force=True)

		if args.test:

			for global_step in range(10):

				x1, y1 = generate_random_wave(num_samples)
				x2, y2 = generate_random_wave(num_samples)

				embedding1 = network.get_embedding(sess, [x1, x2])
				embedding2 = network.get_embedding(sess, [x2])

				print(embedding1, embedding2, (y1 == y2).all(), embedding1.shape)