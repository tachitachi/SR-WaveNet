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
	parser.add_argument('--teacher', type=str, default='teachers/%d' % int(time.time() * 1000), help='Directory where checkpoint and summary is stored')
	parser.add_argument('--student', type=str, default='students/%d' % int(time.time() * 1000), help='Directory where checkpoint and summary is stored')


	parser.add_argument('--train-teacher', action='store_true', help='Train teacher')
	parser.add_argument('--train-student', action='store_true', help='Train student')

	parser.add_argument('--test-teacher-fast', action='store_true', help='Test teacher (fast generation)')
	parser.add_argument('--test-teacher-slow', action='store_true', help='Test teacher (slow generation)')
	parser.add_argument('--test-student', action='store_true', help='Test student')

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
	teacher = WaveNetAutoEncoder(input_size=num_samples, condition_size=num_classes, output_size=quantization_channels, dilations=dilations, pool_stride=512)

	# input_size, condition_size, output_size, dilations, teacher, num_flows=2, filter_width=2, dilation_channels=32, skip_channels=256, 
	# latent_channels=16, pool_stride=512, name='ParallelWaveNet', learning_rate=0.001
	student = ParallelWaveNet(input_size=num_samples, condition_size=num_classes, output_size=quantization_channels, 
		dilations=dilations, teacher=teacher, num_flows=4, pool_stride=512, learning_rate=1e-5)


	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		teacher.load(args.teacher)
		student.load(args.student)

		if args.train_teacher:
			for global_step in range(num_steps):
				#x, y = audio_data.TrainBatch(batch_size)

				x, y = generate_wave_batch(batch_size, num_samples)
				loss = teacher.train(x, y)

				if global_step % print_steps == 0:
					print(global_step, loss)
				# Checkpoint once per minute
				teacher.save(args.teacher, global_step, force=False)

			teacher.save(args.teacher, global_step, force=True)

		if args.train_student:
			for global_step in range(num_steps):
				x, y = generate_wave_batch(batch_size, num_samples)

				encoding = teacher.encode(x, y) 

				noise = np.random.random(x.shape) * 2 - 1

				loss = student.train(noise, y, encoding)

				if True or global_step % print_steps == 0:
					entropy = student.getEntropy(noise, y, encoding)
					print(global_step, loss, entropy)


				student.save(args.student, global_step, force=False)
			student.save(args.student, global_step, force=True)


		if args.test_teacher_fast:
			for global_step in range(10):
				#x, y = audio_data.TrainBatch(batch_size)

				x, y = generate_wave_batch(batch_size, num_samples, combos=True)
				x2, y2 = generate_wave_batch(batch_size, num_samples, combos=True)

				regen = teacher.reconstruct(x, y)
				encoding = teacher.encode(x2, y2) 

				regen2 = teacher.reconstruct_with_encoding(x2, y2, encoding)

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

		if args.test_teacher_slow:
			for count in range(10):

				x, y = generate_wave_batch(batch_size, num_samples, combos=True)
				#x2, y2 = generate_wave_batch(batch_size, num_samples, combos=True)

				encoding = teacher.encode(x, y)

				x_so_far = np.zeros((1, num_samples))



				x_so_far = teacher.reconstruct_with_encoding(x_so_far, y, encoding)

				print(x_so_far, x_so_far.shape)

				for i in range(700):
					x_so_far[:,i:] = 0
					#print(x_so_far[:,:100])
					#print(teacher.get_logits(x_so_far, y, encoding))
					x_so_far[:, i] = teacher.reconstruct_with_encoding(x_so_far, y, encoding)[:, i]


				x_so_far[:,i:] = 0
				regen = x_so_far

				plt.figure(1)
				plt.subplot(211)

				plt.plot(np.arange(num_samples), x[0])

				plt.subplot(212)
				plt.plot(np.arange(num_samples), regen[0])

				plt.show()

