#!python3.10

#   MIT License - Copyright (c) 2023 Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan

import librosa, numpy as np, soundfile as sf

MAX_SPEC = 'Max Spec'
MIN_SPEC = 'Min Spec'
AVERAGE  = 'Average'

def Normalize(audio):
	"""
	Normalize audio to -1.0 dB peak amplitude
	This is mandatory because every process is based on RMS dB levels.
	(Volumes Compensations & audio Substractions)
	"""
	audio = audio.T
	
	# Suppress DC shift (center on 0.0 vertically)
	audio -= np.mean(audio)

	# Normalize audio peak amplitude to -1.0 dB
	max_peak = np.max(np.abs(audio))
	if max_peak > 0.0:
		max_db = 10 ** (-1.0 / 20)  # Convert -1.0 dB to linear scale
		audio /= max_peak
		audio *= max_db

	return audio.T

def Silent(audio_in, sample_rate, threshold_db = -50):
	"""
	Make silent the parts of audio where dynamic range (RMS) goes below threshold.
	Don't misundertand : this function is NOT a noise reduction !
	Its behavior is to clean the audio from "silent parts" (below -XX dB) to :
	- avoid the MLM model to work on "silent parts", and save GPU time
	- avoid the MLM model to produce artifacts on "silent parts"
	- clean the final audio files from residues of "silent parts"
	"""

	min_size		= int(1.000 * sample_rate)  # 1000 ms
	window_frame	= int(0.010 * sample_rate)  #   10 ms
	fade_duration	= int(0.250 * sample_rate)  #  250 ms
	fade_out		= np.linspace(1.0, 0.0, fade_duration)
	fade_in			= np.linspace(0.0, 1.0, fade_duration)

	start = 0; end = 0; audio_length = audio_in.shape[1]
	audio = audio_in.copy()

	for i in range(0, audio_length, window_frame):
		
		# TODO : Maybe use S=audio (Spectrogram) instead of y=audio ??
		RMS = np.max(librosa.amplitude_to_db(librosa.feature.rms(y=audio[:, i:(i + window_frame)], frame_length=window_frame, hop_length=window_frame)))
		
		if RMS < threshold_db:
			end = i + window_frame
			# Last part (in case of silence at the end)
			if i == audio_length - window_frame:
				if end - start > min_size:
					# Fade out
					if start > fade_duration:
						audio[:, start:(start + fade_duration)] *= fade_out
						start += fade_duration

					# Clean in between
					audio[:, start:end] = 0.0
		else:
			# Clean the "min_size" samples found
			if end - start > min_size:

				# Fade out
				if start > fade_duration:
					audio[:, start:(start + fade_duration)] *= fade_out
					start += fade_duration

				# Fade in
				if end < audio_length - fade_duration:
					audio[:, (end - fade_duration):end] *= fade_in
					end -= fade_duration
		
				# Clean in between
				audio[:, start:end] = 0.0

			start = i

	return audio



# - For the code below :
#   MIT License
#
#   Copyright (c) 2023 Anjok07 & aufr33 - Ultimate Vocal Remover (UVR 5)
#
# - https://github.com/Anjok07/ultimatevocalremovergui

def Make_Ensemble(algorithm, audio_input):

	if len(audio_input) == 1:
		return audio_input[0]
	
	waves = []
	
	if algorithm == AVERAGE:
		output = average_audio(audio_input)
	else:
		specs = []
		
		for i in range(len(audio_input)):  
			waves.append(audio_input[i])
			spec = wave_to_spectrogram_no_mp(audio_input[i])
			specs.append(spec)
		
		waves_shapes = [w.shape[1] for w in waves]
		target_shape = waves[waves_shapes.index(max(waves_shapes))]
		
		output = spectrogram_to_wave_no_mp(ensembling(algorithm, specs))
		output = to_shape(output, target_shape.shape)

	return output

def ensembling(a, specs):   
	for i in range(1, len(specs)):
		if i == 1:
			spec = specs[0]

		ln = min([spec.shape[2], specs[i].shape[2]])
		spec = spec[:,:,:ln]
		specs[i] = specs[i][:,:,:ln]
		
		if MIN_SPEC == a:
			spec = np.where(np.abs(specs[i]) <= np.abs(spec), specs[i], spec)
		if MAX_SPEC == a:
			spec = np.where(np.abs(specs[i]) >= np.abs(spec), specs[i], spec)  
		if AVERAGE == a:
			spec = np.where(np.abs(specs[i]) == np.abs(spec), specs[i], spec)  

	return spec

def spectrogram_to_wave_no_mp(spec):
	wave = librosa.istft(spec, n_fft=4096, hop_length=1024)
	
	if wave.ndim == 1:  wave = np.asfortranarray([wave, wave])
	return wave

def wave_to_spectrogram_no_mp(wave):
	spec = librosa.stft(wave, n_fft=4096, hop_length=1024)
	
	if spec.ndim == 1:  spec = np.asfortranarray([spec, spec])
	return spec

def to_shape(x, target_shape):
	padding_list = []
	for x_dim, target_dim in zip(x.shape, target_shape):
		pad_value = (target_dim - x_dim)
		pad_tuple = ((0, pad_value))
		padding_list.append(pad_tuple)
	
	return np.pad(x, tuple(padding_list), mode='constant')

def average_audio(audio_input):
	
	waves = []
	waves_shapes = []
	final_waves = []

	for i in range(len(audio_input)):
		wave = audio_input[i]
		waves.append(wave)
		waves_shapes.append(wave.shape[1])

	wave_shapes_index = waves_shapes.index(max(waves_shapes))
	target_shape = waves[wave_shapes_index]
	waves.pop(wave_shapes_index)
	final_waves.append(target_shape)

	for n_array in waves:
		wav_target = to_shape(n_array, target_shape.shape)
		final_waves.append(wav_target)

	waves = sum(final_waves)
	waves = waves / len(audio_input)

	return waves
