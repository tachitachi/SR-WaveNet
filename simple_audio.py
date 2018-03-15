import numpy as np
import scipy.signal


def generate_random_wave(length, combos=False):
	funcs = [Sine, Square, Sawtooth, Triangle]

	#frequency = 20

	labels = np.zeros(len(funcs))

	if combos:
		num_waves = np.random.randint(1, 5)
	else:
		num_waves = 1

	# Choose which functions to compose together
	choices = np.random.choice(np.arange(len(funcs)), num_waves, replace=False)

	wave = None

	for choice in choices:

		#frequency = np.random.randint(20) + 2
		frequency = 20
		if wave is None:
			wave = funcs[choice](frequency=frequency, duration=1, sample_rate=length)
		else:
			wave = wave + funcs[choice](frequency=frequency, duration=1, sample_rate=length)
		labels[choice] = 1

	# Add in a small amount of gaussian noise
	wave += np.random.normal(0, 0.1, wave.shape)

	wave = Normalize(wave, min_val=-1, max_val=1)

	return wave, labels


def generate_random_wave_f(length, combos=False):
	funcs = [Sine, Square, Sawtooth, Triangle]

	#frequency = 20

	frequency = np.random.randint(18) + 22
	labels = np.zeros(10)
	labels[int(frequency / 2 - 1) - 10] = 1

	if combos:
		num_waves = np.random.randint(1, 5)
	else:
		num_waves = 1

	# Choose which functions to compose together
	choice = np.random.choice(np.arange(len(funcs)))

	#frequency = 20

	wave = funcs[choice](frequency=frequency, duration=1, sample_rate=length)

	# Add in a small amount of gaussian noise
	wave += np.random.normal(0, 0.1, wave.shape)

	wave = Normalize(wave, min_val=-1, max_val=1)

	return wave, labels

def generate_wave_batch(batch_size, length, combos=False):
	x, y = zip(*[generate_random_wave_f(length, combos) for i in range(batch_size)])
	x = np.array(x)
	y = np.array(y)
	return x, y

def CreateTicks(duration, sample_rate):
	ticks = int(sample_rate * duration)
	return np.linspace(0, duration, ticks)

# TODO: Add detune functionality
# pass in frequency, sample rate, duration
def Sine(frequency, duration, sample_rate=11025, detune=0):
	t = CreateTicks(duration, sample_rate)
	return np.sin(t * 2 * np.pi * frequency)

# TODO: Add detune functionality
def Sawtooth(frequency, duration, sample_rate=11025, detune=0):
	t = CreateTicks(duration, sample_rate)
	return scipy.signal.sawtooth(t * 2 * np.pi * frequency)

# TODO: Add detune functionality
def Square(frequency, duration, sample_rate=11025, detune=0):
	t = CreateTicks(duration, sample_rate)
	return scipy.signal.square(t * 2 * np.pi * frequency)

# TODO: Add detune functionality
# triangle wave is sawtooth(t, width=0.5)
def Triangle(frequency, duration, sample_rate=11025, detune=0):
	t = CreateTicks(duration, sample_rate)
	return scipy.signal.sawtooth(t * 2 * np.pi * frequency, width=0.5)


# Returns a piecewise function scaled from 0 to 1
def Envelope(attack, decay, sustain_value, sustain_duration, release, total_duration, sample_rate=11025):
	t = CreateTicks(total_duration, sample_rate)
	vals = np.zeros_like(t)

	attack_idx, decay_idx, sustain_idx, release_idx = np.searchsorted(t, [0, attack, attack + decay, attack + decay + sustain_duration])

	# Linear ramp for attack from 0 to 1
	attack_len = decay_idx - attack_idx
	vals[attack_idx:decay_idx] = np.linspace(0, 1, attack_len)

	# Linear ramp for decay from 1 to sustain_value
	decay_len = sustain_idx - decay_idx
	vals[decay_idx:sustain_idx] = np.linspace(1, sustain_value, decay_len)

	# hold sustain value
	sustain_len = release_idx - sustain_idx
	vals[sustain_idx:release_idx] = sustain_value

	# Exponential decay from sustain_value to 0
	release_len = int(release * sample_rate)
	release_ramp = np.exp(np.linspace(0, -5, release_len) * 0.693) * sustain_value

	release_end = min(release_idx+release_len, len(t))
	release_real_len = max(0, release_end - release_idx)
	vals[release_idx:release_end] = release_ramp[:release_real_len]

	return vals


def Normalize(t, min_val=0, max_val=1):
	current_min = np.min(t)
	current_max = np.max(t)
	current_range = current_max - current_min

	target_range = max_val - min_val
	return ((t - current_min) / current_range) * target_range + min_val




# TODO: white, brown, pink
def Noise():
	pass

# TODO: "lowpass", "highpass", "bandpass", "lowshelf", "highshelf", "notch", "allpass", or "peaking"
def Filter():
	pass

if __name__ == '__main__':
	import matplotlib.pyplot as plt

	frequency = 5
	duration = 1
	sample_rate = 11025

	wave1 = Sine(frequency=frequency, duration=duration, sample_rate=sample_rate)
	wave2 = Sawtooth(frequency=frequency * 3, duration=duration, sample_rate=sample_rate)
	wave3 = Square(frequency=frequency * 2, duration=duration, sample_rate=sample_rate)
	wave4 = Triangle(frequency=frequency, duration=duration, sample_rate=sample_rate)

	env = Envelope(attack=0.1, decay=0.2, sustain_value=0.3, sustain_duration=0.5, release=0.1, total_duration=duration, sample_rate=sample_rate)

	plt.figure(1)
	plt.subplot(321)
	plt.plot(list(range(len(wave1))), wave1)

	plt.subplot(322)
	plt.plot(list(range(len(wave2))), wave2)

	plt.subplot(323)
	plt.plot(list(range(len(wave3))), wave3)

	plt.subplot(324)
	plt.plot(list(range(len(wave4))), wave4)

	plt.subplot(325)
	plt.plot(list(range(len(wave1))), Normalize(wave1 + wave2 + wave3, -1, 2))


	plt.subplot(326)
	plt.plot(list(range(len(wave1))), env * wave1)

	plt.show()
