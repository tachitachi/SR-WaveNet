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
	
	parser.add_argument('--latent-channels', type=int, default=32, help='Number of latent channel per time slice')
	parser.add_argument('--pool-stride', type=int, default=128, help='Number of samples to use per time slice')

	parser.add_argument('--batch-size', type=int, default=4, help='Batch size')

	parser.add_argument('--entropy-weight', type=float, default=0.25, help='Weight of entropy term in loss function')
	parser.add_argument('--cross-entropy-weight', type=float, default=1.0, help='Weight of cross entropy term in loss function')
	parser.add_argument('--power-weight', type=float, default=1.0, help='Weight of power loss term in loss function')
	parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')

	args = parser.parse_args()

	batch_size = args.batch_size
	num_steps = 1000000
	print_steps = 25
	use_condition = False

	sample_rate = 4000
	num_samples = 4096 # 16384
	num_classes = 128 if use_condition else 0

	#audio_data = AudioData()
	#audio_data = NsynthDataReader(os.path.join('nsynth_data', 'nsynth-train.tfrecord'), batch_size)
	#audio_data = NsynthDataReader(os.path.join('nsynth_data', 'synthetic_valid.tfrecord'), batch_size, num_samples)
	#audio_data = NsynthDataReader(os.path.join('nsynth_data', 'filtered_note60.tfrecord'), batch_size, num_samples)
	audio_data = NsynthDataReader(os.path.join('nsynth_data', 'filtered_note60_4000.tfrecord'), batch_size, num_samples, audio_max_length=16000)
	dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
              1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
              1, 2, 4, 8, 16, 32, 64, 128, 256, 512]

	entropy_weight = args.entropy_weight
	cross_entropy_weight = args.cross_entropy_weight
	power_weight = args.power_weight
	
	latent_channels = args.latent_channels # 16
	pool_stride = args.pool_stride # 512

    # input_size, condition_size, output_size, dilations, filter_width=2, encoder_channels=128, dilation_channels=32, skip_channels=256, 
	# output_channels=256, latent_channels=16, pool_stride=512, name='WaveNetAutoEncoder', learning_rate=0.001):
	#teacher = WaveNetAutoEncoder(input_size=num_samples, condition_size=num_classes, num_mixtures=5, dilations=dilations, pool_stride=512)

	#print('after teacher')

	# input_size, condition_size, output_size, dilations, teacher, num_flows=2, filter_width=2, dilation_channels=32, skip_channels=256, 
	# latent_channels=16, pool_stride=512, name='ParallelWaveNet', learning_rate=0.001
	student = ParallelWaveNet(input_size=num_samples, condition_size=num_classes,
		dilations=dilations, teacher=args.teacher, dilation_channels=32, skip_channels=128, num_flows=4, 
		latent_channels=latent_channels, pool_stride=pool_stride, 
		alpha=entropy_weight, beta=cross_entropy_weight, gamma=power_weight, learning_rate=args.learning_rate)


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
				if not use_condition:
					y = None

				encoding = student.encode(sess, x, y) 

				num_random_samples = batch_size

				#encoding = np.tile(encoding, [num_random_samples, 1, 1])
				#y_stack = np.tile(y, [num_random_samples, 1]) if y else None
				y_stack = y

				#noise = np.random.logistic(0, 1, [num_random_samples, num_samples])
				noise = np.random.logistic(0, 1, [batch_size, num_samples])

				# Train multiple times on different samples
				loss, power_loss = student.train_fast(sess, noise, x, encoding, y_stack)

				if global_step % print_steps == 0:
					#teacher_logits = teacher.
					entropy = student.getEntropy_fast(sess, noise, encoding, y_stack)
					print('Step: {:6d} | Entropy: {} | Power Loss: {:.4f} | Total Loss: {:.4f}'.format(global_step, str(entropy), power_loss, loss))

					output = student.generate(sess, noise, encoding, y_stack)

					regen = student.reconstruct(sess, x, y)

					plt_count = 1
					plt.figure(1, figsize=(10, 8))
					for i in range(batch_size):
						plt.subplot(batch_size, 5, plt_count)
						plt.plot(np.arange(num_samples), x[i])
						plt_count += 1
						
						plt.subplot(batch_size, 5, plt_count)
						plt.plot(np.arange(num_samples), regen[i])
						plt_count += 1
						
						plt.subplot(batch_size, 5, plt_count)
						plt.plot(np.arange(num_samples), noise[i])
						plt_count += 1
						
						plt.subplot(batch_size, 5, plt_count)
						plt.plot(np.arange(num_samples), output[i])
						plt_count += 1
						
						plt.subplot(batch_size, 5, plt_count)
						plt.imshow(encoding[i].transpose())
						plt_count += 1


					if not os.path.isdir(os.path.join(args.student, 'figures')):
						os.makedirs(os.path.join(args.student, 'figures'))

					plt.savefig(os.path.join(args.student, 'figures', '{}.png'.format(global_step)))

					plt.close()

					if global_step % 200 == 0:
						if not os.path.isdir(os.path.join(args.student, 'audio')):
							os.makedirs(os.path.join(args.student, 'audio'))

						wavfile.write(os.path.join(args.student, 'audio', 'test_wav_{}.wav'.format(global_step)), sample_rate, x[0])
						wavfile.write(os.path.join(args.student, 'audio', 'regen_wav_{}.wav'.format(global_step)), sample_rate, regen[0])
						wavfile.write(os.path.join(args.student, 'audio', 'parallel_wav_{}.wav'.format(global_step)), sample_rate, output[0])



				student.save(sess, args.student, global_step, force=False)
			student.save(sess, args.student, global_step, force=True)


		if args.test:
			for global_step in range(20):
				x, y = audio_data.next()
				if not use_condition:
					y = None

				encoding = student.encode(sess, x, y) 
				regen = student.reconstruct(sess, x, y)

				noise = np.random.logistic(0, 1, x.shape)
				entropy = student.getEntropy(sess, noise, encoding, y)
				output = student.generate(sess, noise, encoding, y)

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

