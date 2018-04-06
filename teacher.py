import argparse
import time
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import WaveNetAutoEncoder

from nsynth import NsynthDataReader

from scipy.io import wavfile

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--teacher', type=str, default='teachers/%d' % int(time.time() * 1000), help='Directory where checkpoint and summary is stored')
	parser.add_argument('--student', type=str, default='students/%d' % int(time.time() * 1000), help='Directory where checkpoint and summary is stored')
	parser.add_argument('--start', type=int, default=0, help='Starting index')

	parser.add_argument('--train', action='store_true', help='Train teacher')

	parser.add_argument('--test-fast', action='store_true', help='Test teacher (fast generation)')
	parser.add_argument('--test-slow', action='store_true', help='Test teacher (slow generation)')


	parser.add_argument('--latent-channels', type=int, default=32, help='Number of latent channel per time slice')
	parser.add_argument('--pool-stride', type=int, default=128, help='Number of samples to use per time slice')

	parser.add_argument('--batch-size', type=int, default=4, help='Batch size')

	args = parser.parse_args()

	batch_size = args.batch_size
	num_steps = 1000000
	print_steps = 100

	use_condition = False

	last_checkpoint_time = time.time()

	num_samples = 4096 # 16384
	num_classes = 128 if use_condition else 0

	latent_channels = args.latent_channels # 16
	pool_stride = args.pool_stride # 512

	#audio_data = AudioData()
	#audio_data = NsynthDataReader(os.path.join('nsynth_data', 'nsynth-train.tfrecord'), batch_size)
	#audio_data = NsynthDataReader(os.path.join('nsynth_data', 'synthetic_valid.tfrecord'), batch_size, num_samples)
	#audio_data = NsynthDataReader(os.path.join('nsynth_data', 'filtered_note60.tfrecord'), batch_size, num_samples)
	audio_data = NsynthDataReader(os.path.join('nsynth_data', 'filtered_note60_4000.tfrecord'), batch_size, num_samples, audio_max_length=16000)

	dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
              1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
              1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    # input_size, condition_size, num_mixtures, dilations, filter_width=2, encoder_channels=128, dilation_channels=32, skip_channels=256, 
	# latent_channels=16, pool_stride=512, name='WaveNetAutoEncoder', learning_rate=0.001)
	teacher = WaveNetAutoEncoder(input_size=num_samples, condition_size=num_classes, num_mixtures=5, dilations=dilations, 
		latent_channels=latent_channels, skip_channels=128, pool_stride=pool_stride, learning_rate=1e-4)

	with tf.Session(graph=teacher.graph) as sess:
		sess.run(tf.global_variables_initializer())

		teacher.load(args.teacher)


		if args.train:
			for global_step in range(args.start, num_steps):

				x, y = audio_data.next()

				if not use_condition:
					y = None

				loss = teacher.train(x, y)

				if global_step % print_steps == 0:
					print(global_step, loss)

					regen = teacher.reconstruct(x, y)
					encoding = teacher.encode(x, y)

					plt.figure(1, figsize=(10, 8))
					plt.subplot(3, 1, 1)
					plt.plot(np.arange(num_samples), x[0])

					plt.subplot(3, 1, 2)
					plt.plot(np.arange(num_samples), regen[0])

					plt.subplot(3, 1, 3)
					plt.imshow(encoding[0].transpose())

					if not os.path.isdir(os.path.join(args.teacher, 'figures')):
						os.makedirs(os.path.join(args.teacher, 'figures'))

					plt.savefig(os.path.join(args.teacher, 'figures', '{}.png'.format(global_step)))

					plt.close()


					if global_step % 500 == 0:
						if not os.path.isdir(os.path.join(args.teacher, 'audio')):
							os.makedirs(os.path.join(args.teacher, 'audio'))

						wavfile.write(os.path.join(args.teacher, 'audio', 'test_wav_{}.wav'.format(global_step)), 16000, x[0])
						wavfile.write(os.path.join(args.teacher, 'audio', 'regen_wav_{}.wav'.format(global_step)), 16000, regen[0])

				# Checkpoint once per minute
				teacher.save(args.teacher, global_step, force=False)

			teacher.save(args.teacher, global_step, force=True)


		if args.test_fast:
			for global_step in range(10):
				x, y = audio_data.next()

				if not use_condition:
					y = None

				regen = teacher.reconstruct(x, y)

				wavfile.write('test_wav_{}.wav'.format(global_step), 16000, x[0])
				wavfile.write('regen_wav_{}.wav'.format(global_step), 16000, regen[0])


				plt.figure(1)
				plt.subplot(211)

				plt.plot(np.arange(num_samples), x[0])

				plt.subplot(212)
				plt.plot(np.arange(num_samples), regen[0])

				plt.show()

		if args.test_slow:
			for count in range(1):

				x, y = audio_data.next()

				if not use_condition:
					y = None
				#x2, y2 = generate_wave_batch(batch_size, num_samples, combos=True)

				wavfile.write('test_wav_{}.wav'.format(count), 16000, x[0])

				encoding = teacher.encode(x, y)

				x_so_far = np.zeros((1, num_samples))



				x_so_far = teacher.reconstruct_with_encoding(x_so_far, encoding, y)

				print(x_so_far, x_so_far.shape)

				for i in range(num_samples):
					if i % 100 == 0:
						print('{} out of {}'.format(i, num_samples))
					x_so_far[:,i:] = 0
					#print(x_so_far[:,:100])
					#print(teacher.get_logits(x_so_far, y, encoding))
					x_so_far[:, i] = teacher.reconstruct_with_encoding(x_so_far, encoding)[:, i]


				x_so_far[:,i:] = 0
				regen = x_so_far


				wavfile.write('regen_wav_{}.wav'.format(count), 16000, regen[0])

				plt.figure(1)
				plt.subplot(211)

				plt.plot(np.arange(num_samples), x[0])

				plt.subplot(212)
				plt.plot(np.arange(num_samples), regen[0])

				plt.show()

