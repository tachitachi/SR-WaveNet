# Module for getting audio data frames
import scipy.io.wavfile as wavfile
import os
import re
import numpy as np

class AudioData(object):
	def __init__(self, datadir='data', num_samples=16000):
		self.datadir = datadir
		self.num_samples = num_samples

		self.train_files = {}
		self.test_files = {}
		self.validation_files = {}

		self.labelToIdx = {}
		self.idxToLabel = {}

		# Get test list
		with open(os.path.join(self.datadir, 'testing_list.txt'), 'r') as f:
			for line in f.readlines():
				path = line.strip()
				label, _ = line.split('/')
				fullpath = os.path.join(self.datadir, path)
				self.test_files[fullpath] = {'data': None, 'label': label}

				if label not in self.labelToIdx:
					idx = len(self.labelToIdx)
					self.labelToIdx[label] = idx
					self.idxToLabel[idx] = label

		# Get validation list
		with open(os.path.join(self.datadir, 'validation_list.txt'), 'r') as f:
			for line in f.readlines():
				path = line.strip()
				label, _ = line.split('/')
				fullpath = os.path.join(self.datadir, path)
				self.validation_files[fullpath] = {'data': None, 'label': label}

				if label not in self.labelToIdx:
					idx = len(self.labelToIdx)
					self.labelToIdx[label] = idx
					self.idxToLabel[idx] = label

		# Create refs to wav files
		for root, dirs, files in os.walk(self.datadir):
			if root == self.datadir:
				# skip files immediately in data directory
				continue

			expr = re.escape(self.datadir) + re.escape(os.path.sep)  + '[^' + re.escape(os.path.sep) + ']+'
			if not re.search(expr, root):
				continue

			label = root.split(os.path.sep)[-1]

			# Remove background noise, and other helper directories
			if label.startswith('_'):
				continue

			# Hopefully nothing is new here, but just in case
			if label not in self.labelToIdx:
				idx = len(self.labelToIdx)
				self.labelToIdx[label] = idx
				self.idxToLabel[idx] = label

			for file in files:
				fullpath = os.path.join(root, file)

				# Only add training files
				if fullpath in self.test_files or fullpath in self.validation_files:
					continue

				self.train_files[fullpath] = {'data': None, 'label': label}

	@property
	def classes(self):
		return len(self.labelToIdx)

	def Load(self, fullpath, files):
		data, label = None, None

		if fullpath in files:
			filedata = files[fullpath]
			if filedata['data'] is None:
				r, d = wavfile.read(fullpath)
				filedata['data'] = d
			data = filedata['data']
			label = filedata['label']

		if data.shape[0] < self.num_samples:
			# pad end with 0s
			difference = self.num_samples - data.shape[0]
			data = np.pad(data, [0, difference], 'constant')

		return data[:self.num_samples], self.labelToIdx[label]


	def _GetBatch(self, batch_size, files):
		fullpaths = np.random.choice(list(files.keys()), batch_size)
		data, labels = zip(*[self.Load(p, files) for p in fullpaths])

		data = np.stack(data, axis=0) / 32767

		one_hot = np.zeros((batch_size, len(self.labelToIdx)))
		labels = np.array(labels, dtype=np.int32)
		one_hot[np.arange(batch_size), labels] = 1

		return data, one_hot

	def TrainBatch(self, batch_size):
		return self._GetBatch(batch_size, self.train_files)

	def TestBatch(self, batch_size):
		return self._GetBatch(batch_size, self.test_files)

	def ValidationBatch(self, batch_size):
		return self._GetBatch(batch_size, self.validation_files)


if __name__ == '__main__':
	audio = AudioData()

	x, y = audio.TrainBatch(16)
	print(x.shape, y.shape)

	x, y = audio.TestBatch(12)
	print(x.shape, y.shape)

	x, y = audio.ValidationBatch(8)
	print(x.shape, y.shape)