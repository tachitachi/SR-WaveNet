import argparse
import time
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data import AudioData
from model import WaveNetAutoEncoder, ParallelWaveNet

from simple_audio import generate_wave_batch

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--logdir', type=str, default='events/%d' % int(time.time() * 1000), help='Directory where checkpoint and summary is stored')
	parser.add_argument('--test', action='store_true', help='Test mode')
	parser.add_argument('--gen', action='store_true', help='Generate mode')
	parser.add_argument('--parallel', action='store_true', help='parallel generation mode')
	args = parser.parse_args()

	batch_size = 1
	num_steps = 100000
	print_steps = 100

	last_checkpoint_time = time.time()

	audio_data = AudioData()
	num_samples = audio_data.num_samples
	num_classes = audio_data.classes

	quantization_channels = 256

	num_samples = 5120
	num_classes = 10
	quantization_channels = 256


	dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
              1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
              1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    # input_size, condition_size, output_size, dilations, filter_width=2, encoder_channels=128, dilation_channels=32, skip_channels=256, 
	# output_channels=256, latent_channels=16, pool_stride=512, name='WaveNetAutoEncoder', learning_rate=0.001):
	network = WaveNetAutoEncoder(input_size=num_samples, condition_size=num_classes, output_size=quantization_channels, dilations=dilations, pool_stride=512)

	# input_size, condition_size, output_size, dilations, teacher, num_flows=2, filter_width=2, dilation_channels=32, skip_channels=256, 
	# latent_channels=16, pool_stride=512, name='ParallelWaveNet', learning_rate=0.001
	student = ParallelWaveNet(input_size=num_samples, condition_size=num_classes, output_size=quantization_channels, 
		dilations=dilations, teacher=network, num_flows=4, pool_stride=512)

	saver = tf.train.Saver(network.network_params)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())


		if args.logdir is not None and os.path.exists(args.logdir):
			checkpoint_state = tf.train.get_checkpoint_state(args.logdir)
			if checkpoint_state is not None:
				try:
					saver.restore(sess, checkpoint_state.model_checkpoint_path)
					print('Restoring previous session')
				except (tf.errors.NotFoundError):
					print('Could not find checkpoint at %s', checkpoint_state.model_checkpoint_path)

		if not args.test:

			for global_step in range(num_steps):
				#x, y = audio_data.TrainBatch(batch_size)

				x, y = generate_wave_batch(batch_size, num_samples, combos=True)

				loss = network.train(x, y)


				if global_step % print_steps == 0:
					print(global_step, loss)

#					regen = network.reconstruct(x, y)
#
#					plt.figure(1)
#					plt.subplot(211)
#
#					plt.plot(np.arange(num_samples), x[0])
#
#					plt.subplot(212)
#					plt.plot(np.arange(num_samples), regen[0])
#
#					plt.show()

				# Checkpoint once per minute
				if time.time() - last_checkpoint_time > 60:
					if not os.path.isdir(args.logdir):
						os.makedirs(args.logdir)
					saver.save(sess, os.path.join(args.logdir, 'model.ckpt'), global_step)
					last_checkpoint_time = time.time()

			saver.save(sess, os.path.join(args.logdir, 'model.ckpt'), global_step)

		else:


			if args.parallel:

				for i in range(10):
					x, y = generate_wave_batch(batch_size, num_samples, combos=True)

					encoding = network.encode(x, y) 

					regen = network.reconstruct_with_encoding(x, y, encoding)



					noise1 = np.random.random(x.shape)
					noise2 = np.random.random(x.shape)
					parallel_gen1 = student.generate(noise1, y, encoding)
					parallel_gen2 = student.generate(noise2, y, encoding)


					plt.figure(1)
					plt.subplot(221)

					plt.plot(np.arange(num_samples), x[0])

					plt.subplot(222)
					plt.plot(np.arange(num_samples), regen[0])


					plt.subplot(223)
					plt.plot(np.arange(num_samples), parallel_gen1[0])

					plt.subplot(224)
					plt.plot(np.arange(num_samples), parallel_gen2[0])

					plt.show()


			elif args.gen:


				for count in range(10):

					x, y = generate_wave_batch(batch_size, num_samples, combos=True)
					#x2, y2 = generate_wave_batch(batch_size, num_samples, combos=True)

					encoding = network.encode(x, y)

					x_so_far = np.zeros((1, num_samples))



					x_so_far = network.reconstruct_with_encoding(x_so_far, y, encoding)

					print(x_so_far, x_so_far.shape)

					for i in range(700):
						x_so_far[:,i:] = 0
						#print(x_so_far[:,:100])
						#print(network.get_logits(x_so_far, y, encoding))
						x_so_far[:, i] = network.reconstruct_with_encoding(x_so_far, y, encoding)[:, i]


					x_so_far[:,i:] = 0
					regen = x_so_far

					plt.figure(1)
					plt.subplot(211)

					plt.plot(np.arange(num_samples), x[0])

					plt.subplot(212)
					plt.plot(np.arange(num_samples), regen[0])

					plt.show()

					#for i in range(num_samples):

			else:

				for global_step in range(10):
					#x, y = audio_data.TrainBatch(batch_size)

					x, y = generate_wave_batch(batch_size, num_samples, combos=True)
					x2, y2 = generate_wave_batch(batch_size, num_samples, combos=True)

					regen = network.reconstruct(x, y)
					encoding = network.encode(x2, y2) 

					regen2 = network.reconstruct_with_encoding(x2, y2, encoding)

					plt.figure(1)
					plt.subplot(221)

					plt.plot(np.arange(num_samples), x[0])

					plt.subplot(222)
					plt.plot(np.arange(num_samples), x2[0])

					plt.subplot(223)
					plt.plot(np.arange(num_samples), regen[0])

					plt.subplot(224)
					plt.plot(np.arange(num_samples), regen2[0])

					plt.show()