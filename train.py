import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('--classifier', action='store_true', help='Train a classifier wavenet')
	parser.add_argument('--siamese', action='store_true', help='Train a siamese wavenet')
	args = parser.parse_args()

	if args.classifier:

		pass

	if args.siamese:

		pass