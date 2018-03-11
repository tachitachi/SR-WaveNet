import argparse
import time
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from data import AudioData
from model import WaveNet

#from simple_audio import generate_wave_batch


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--logdir', type=str, default='events/%d' % int(time.time() * 1000), help='Directory where checkpoint and summary is stored')
	parser.add_argument('--classifier', action='store_true', help='Train a classifier wavenet')
	parser.add_argument('--siamese', action='store_true', help='Train a siamese wavenet')
	parser.add_argument('--test', action='store_true', help='Test mode')
	args = parser.parse_args()

	batch_size = 1
	num_steps = 100000
	print_steps = 100

	last_checkpoint_time = time.time()

	if args.classifier:

		audio_data = AudioData()
		num_samples = audio_data.num_samples
		num_classes = audio_data.classes

		dilations = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                  1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


		network = WaveNet(num_samples, num_classes, dilations, dilation_channels=32, skip_channels=128, output_channels=num_classes, learning_rate=0.001)

		saver = tf.train.Saver()

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
					x, y = audio_data.TrainBatch(batch_size)
					loss = network.train(x, y)

					if global_step % print_steps == 0:
						labels = network.predict(x)
						#print(loss, np.sum(np.abs(y - np.round(labels))))
						print('Step: {:6d} | Loss: {:.4f} | Correct Pred: {:.4f} | Real: {} | Predicted: {}'.format(global_step, loss, np.sum(y * labels), 
							audio_data.getWord(np.argmax(y)), audio_data.getWord(np.argmax(labels))))


					# Checkpoint once per minute
					if time.time() - last_checkpoint_time > 60:
						if not os.path.isdir(args.logdir):
							os.makedirs(args.logdir)
						saver.save(sess, os.path.join(args.logdir, 'model.ckpt'), global_step)
						last_checkpoint_time = time.time()

				saver.save(sess, os.path.join(args.logdir, 'model.ckpt'), global_step)
			else:


				if False:
					for global_step in range(num_steps):
						x, y = audio_data.TestBatch(1)
						labels = network.predict(x)

						print('Step: {:6d} | Correct Pred: {:.4f} | Real: {} | Predicted: {}'.format(global_step, np.sum(y * labels), 
							audio_data.getWord(np.argmax(y)), audio_data.getWord(np.argmax(labels))))


				if True:
					corrects = {}
					wrongs = {}
					correct_count = 0
					wrong_count = 0
					for filepath in audio_data.test_files:
						data, label = audio_data.Load(filepath, audio_data.test_files)

						x = np.array([data])


						labels = network.predict(x)

						pred_word = audio_data.idxToLabel[np.argmax(labels)]
						real_word = audio_data.idxToLabel[label]

						if real_word not in corrects:
							corrects[real_word] = 0
						if real_word not in wrongs:
							wrongs[real_word] = 0

						if np.argmax(labels) == label:
							corrects[real_word] += 1
							correct_count += 1
						else:
							wrongs[real_word] += 1
							wrong_count += 1

					print('Correct: {:d} | Wrong: {:d} | Total: {:d} | % Correct: {:.3f}'.format(correct_count, wrong_count, correct_count + wrong_count, 
						correct_count / float(correct_count + wrong_count)))

					print('Correct', corrects)
					print('Wrongs', wrongs)


	if args.siamese:

		pass