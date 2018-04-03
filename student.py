import argparse
import time
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.io import wavfile

#from data import AudioData
from model import ParallelWaveNet

#from simple_audio import generate_wave_batch
from nsynth import NsynthDataReader

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--teacher', type=str, default='teachers/%d' % int(time.time() * 1000), help='Directory where checkpoint and summary is stored')
	parser.add_argument('--student', type=str, default='students/%d' % int(time.time() * 1000), help='Directory where checkpoint and summary is stored')
	parser.add_argument('--start', type=int, default=0, help='Starting index')

	parser.add_argument('--train', action='store_true', help='Train student')
	parser.add_argument('--test', action='store_true', help='Test student')

	args = parser.parse_args()

	batch_size = 1
	num_steps = 1000000
	print_steps = 25

	num_samples = 16384
	num_classes = 128

	#audio_data = AudioData()
	#audio_data = NsynthDataReader(os.path.join('nsynth_data', 'nsynth-train.tfrecord'), batch_size)
	audio_data = NsynthDataReader(os.path.join('nsynth_data', 'synthetic_valid.tfrecord'), batch_size, num_samples)
	dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
              1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
              1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

    # input_size, condition_size, output_size, dilations, filter_width=2, encoder_channels=128, dilation_channels=32, skip_channels=256, 
	# output_channels=256, latent_channels=16, pool_stride=512, name='WaveNetAutoEncoder', learning_rate=0.001):
	#teacher = WaveNetAutoEncoder(input_size=num_samples, condition_size=num_classes, num_mixtures=5, dilations=dilations, pool_stride=512)

	#print('after teacher')

	# input_size, condition_size, output_size, dilations, teacher, num_flows=2, filter_width=2, dilation_channels=32, skip_channels=256, 
	# latent_channels=16, pool_stride=512, name='ParallelWaveNet', learning_rate=0.001
	student = ParallelWaveNet(input_size=num_samples, condition_size=num_classes,
		dilations=dilations, teacher=args.teacher, dilation_channels=32, skip_channels=128, num_flows=4, pool_stride=512, learning_rate=1e-4)


	with tf.Session(graph=student.graph) as sess:
		sess.run(tf.global_variables_initializer())

		print('initailized')

		#teacher.load(args.teacher)
		student.load(sess, args.student)

		print('loaded')

		#print('after load')


		if args.train:
			for global_step in range(args.start, num_steps):
				x, y = audio_data.next()

				encoding = student.encode(sess, x, y) 

				num_random_samples = 3

				encoding = np.tile(encoding, [num_random_samples, 1, 1])
				y_stack = np.tile(y, [num_random_samples, 1])

				noise = np.random.logistic(0, 1, [num_random_samples, num_samples])

				# Train multiple times on different samples
				loss, power_loss = student.train_fast(sess, noise, y_stack, encoding, x)

				if global_step % print_steps == 0:
					#teacher_logits = teacher.
					entropy = student.getEntropy_fast(sess, noise, y_stack, encoding)
					print('Step: {:6d} | Entropy: {} | Power Loss: {:.4f} | Total Loss: {:.4f}'.format(global_step, str(entropy), power_loss, loss))

					output = student.generate(sess, noise, y_stack, encoding)

					regen = student.reconstruct(sess, x, y)

					plt.figure(1, figsize=(10, 8))
					plt.subplot(4, 2, 1)
					plt.plot(np.arange(num_samples), x[0])

					plt.subplot(4, 2, 2)
					plt.plot(np.arange(num_samples), regen[0])

					plt.subplot(4, 2, 3)
					plt.plot(np.arange(num_samples), noise[0])

					plt.subplot(4, 2, 4)
					plt.plot(np.arange(num_samples), output[0])


					plt.subplot(4, 2, 5)
					plt.plot(np.arange(num_samples), noise[1])

					plt.subplot(4, 2, 6)
					plt.plot(np.arange(num_samples), output[1])


					plt.subplot(4, 2, 7)
					plt.plot(np.arange(num_samples), noise[2])

					plt.subplot(4, 2, 8)
					plt.plot(np.arange(num_samples), output[2])

					if not os.path.isdir(os.path.join(args.student, 'figures')):
						os.makedirs(os.path.join(args.student, 'figures'))

					plt.savefig(os.path.join(args.student, 'figures', '{}.png'.format(global_step)))

					plt.close()

					if global_step % 200 == 0:
						if not os.path.isdir(os.path.join(args.student, 'audio')):
							os.makedirs(os.path.join(args.student, 'audio'))

						wavfile.write(os.path.join(args.student, 'audio', 'test_wav_{}.wav'.format(global_step)), 16000, x[0])
						wavfile.write(os.path.join(args.student, 'audio', 'regen_wav_{}.wav'.format(global_step)), 16000, regen[0])
						wavfile.write(os.path.join(args.student, 'audio', 'parallel_wav_{}.wav'.format(global_step)), 16000, output[0])



				student.save(sess, args.student, global_step, force=False)
			student.save(sess, args.student, global_step, force=True)


		if args.test:
			for global_step in range(20):
				x, y = audio_data.next()

				encoding = student.encode(sess, x, y) 
				regen = student.reconstruct(sess, x, y)

				noise = np.random.logistic(0, 1, x.shape)
				entropy = student.getEntropy(sess, noise, y, encoding)
				output = student.generate(sess, noise, y, encoding)

				#loss = student.train(sess, noise, y, encoding)

				print('Entropy', entropy)
				#print('loss', loss)


				plt.figure(1)
				plt.subplot(221)

				plt.plot(np.arange(num_samples), x[0])	

				plt.subplot(222)
				plt.plot(np.arange(num_samples), regen[0])

				
				plt.subplot(223)
				plt.plot(np.arange(num_samples), noise[0])
				
				plt.subplot(224)
				plt.plot(np.arange(num_samples), output[0])

				plt.show()

