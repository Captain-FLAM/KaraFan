#!python3.10

#   MIT License - Copyright (c) 2023 - Captain FLAM & Jarredou
#
#   https://github.com/Captain-FLAM/KaraFan

import librosa, numpy as np
from scipy import signal

def Normalize(audio):
	"""
	Normalize audio to -1.0 dB peak amplitude
	This is mandatory for SOME audio files because every process is based on RMS dB levels.
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

	start = 0; end = 0
	audio = audio_in.copy()
	audio_length = audio_in.shape[1]

	for i in range(0, audio_length, window_frame):
		
		# TODO : Maybe use S=audio (Spectrogram) instead of y=audio ??
		RMS = np.max(librosa.amplitude_to_db(librosa.feature.rms(y=audio[:, i:(i + window_frame)], frame_length=window_frame, hop_length=window_frame)))
		# print(f"RMS : {RMS}")
		if RMS < threshold_db:
			end = i + window_frame
			# Last part (in case of silence at the end)
			if i >= audio_length - window_frame:
				if end - start > min_size:
					# Fade out
					if start > fade_duration:
						audio[:, start:(start + fade_duration)] *= fade_out
						start += fade_duration

					# Clean last part
					audio[:, start:audio_length] = 0.0
					break
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

# Linkwitz-Riley filter
#
# Avec cutoff = 17.4khz & -80dB d'atténuation:
#
# ordre =  4 => filtre target freq = 10500hz
# ordre =  6 => filtre target freq = 13200hz
# ordre =  8 => filtre target freq = 14300hz
# ordre = 10 => filtre target freq = 15000hz
# ordre = 12 => filtre target freq = 15500hz
# ordre = 14 => filtre target freq = 15800hz
# ordre = 16 => filtre target freq = 16100hz
#
# Avec cutoff = 17.4khz & -60dB d'atténuation:
#
# ordre =  4 => filtre target freq = 12500hz (-4900)
# ordre =  6 => filtre target freq = 14400hz
# ordre =  8 => filtre target freq = 15200hz (-2200)
# ordre = 10 => filtre target freq = 15700hz
# ordre = 12 => filtre target freq = 16000hz (-1640)
# ordre = 14 => filtre target freq = 16200hz
# ordre = 16 => filtre target freq = 16400hz

def Linkwitz_Riley_filter(type, cutoff, audio, sample_rate, order=8):
	
	# cutoff -= 2200

	nyquist = 0.5 * sample_rate
	normal_cutoff = cutoff / nyquist

	sos = signal.butter(order // 2, normal_cutoff, btype=type, analog=False, output='sos')
	filtered_audio = signal.sosfiltfilt(sos, audio, padlen=0, axis=1)

	return filtered_audio

# Band Pass filter
#
# Vocals -> lowest : 85 - 100 Hz, highest : 20 KHz
# Music  -> lowest : 30 -  50 Hz, highest : 18-20 KHz
#
# Voix masculine :
#
# Minimale : 85 Hz
# Fondamentale : 180 Hz
# Maximale (y compris les harmoniques) : 14 kHz
#
# Voix féminine :
#
# Minimale : 165 Hz
# Fondamentale : 255 Hz
# Maximale (y compris les harmoniques) : 16 kHz
#
# Voix d'enfants :
#
# Minimale : 250 Hz
# Fondamentale : 400 Hz
# Maximale (y compris les harmoniques) : 20 kHz ou +

def Pass_filter(type, cutoff, audio, sample_rate, order=32):

	if cutoff >= sample_rate / 2:
		cutoff = (sample_rate / 2) - 1

	sos = signal.butter(order // 2, cutoff, btype=type, fs=sample_rate, output='sos')
	filtered_audio = signal.sosfiltfilt(sos, audio, padlen=0, axis=1)

	return filtered_audio

# SRS : Sample Rate Scaling
def Change_sample_rate(audio, way, current_cutoff, target_cutoff):

	if way == "DOWN":
		current_cutoff, target_cutoff = target_cutoff, current_cutoff

	pitched_audio = librosa.resample(audio, orig_sr = current_cutoff * 2, target_sr = target_cutoff * 2, res_type = 'kaiser_best', axis=1)

	# print(f"SRS input audio shape: {audio.shape}")
	# print(f"SRS output audio shape: {pitched_audio.shape}")
	# print (f"ratio : {ratio}")

	return pitched_audio

# def Remove_High_freq_Noise(audio, threshold_freq):

# 	# Calculer la transformée de Fourier
# 	stft = librosa.stft(audio)
	
# 	# Calculer la somme des amplitudes pour chaque fréquence dans le spectre
# 	amplitude_sum = np.sum(np.abs(stft), axis=0)

# 	# Appliquer un masque pour supprimer les fréquences supérieures lorsque la somme des amplitudes est inférieure au seuil
# 	stft[:, amplitude_sum > threshold_freq] = 0.0

# 	# Reconstruire l'audio à partir du STFT modifié
	# 	audio_filtered = librosa.istft(stft)

# 	return audio_filtered

# def Match_Freq_CutOFF(self, audio1, audio2, sample_rate):
# 	# This option matches the Primary stem frequency cut-off to the Secondary stem frequency cut-off
# 	# (if the Primary stem frequency cut-off is lower than the Secondary stem frequency cut-off)

# 	# Get the Primary stem frequency cut-off
# 	freq_cut_off1 = Find_Cut_OFF(audio1, sample_rate)
# 	freq_cut_off2 = Find_Cut_OFF(audio2, sample_rate)

# 	# Match the Primary stem frequency cut-off to the Secondary stem frequency cut-off
# 	if freq_cut_off1 < freq_cut_off2:
# 		audio1 = Resize_Freq_CutOFF(audio1, freq_cut_off2, sample_rate)

# 	return audio1

# # Find the high cut-off frequency of the input audio
# def Find_Cut_OFF(audio, sample_rate, threshold=0.01):

# 	# Appliquer un filtre passe-bas pour réduire le bruit
# 	cutoff_frequency = sample_rate / 2.0  # Fréquence de Nyquist (la moitié du taux d'échantillonnage)

# 	# Définir l'ordre du filtre passe-bas
# 	order = 6

# 	# Calculer les coefficients du filtre passe-bas
# 	b, a = signal.butter(order, cutoff_frequency - threshold, btype='low', analog=False, fs=sample_rate)

# 	# Appliquer le filtre au signal audio
# 	filtered_audio = signal.lfilter(b, a, audio, axis=0)

# 	# Calculer la FFT du signal audio filtré
# 	fft_result = np.fft.fft(filtered_audio, axis=0)

# 	# Calculer les magnitudes du spectre de fréquence
# 	magnitudes = np.abs(fft_result)

# 	# Calculer les fréquences correspondant aux bins de la FFT
# 	frequencies = np.fft.fftfreq(len(audio), 1.0 / sample_rate)

# 	# Trouver la fréquence de coupure où la magnitude tombe en dessous du seuil
# 	cut_off_frequencies = frequencies[np.where(magnitudes > threshold)]

# 	# Trouver la fréquence de coupure maximale parmi toutes les valeurs
# 	return int(max(cut_off_frequencies))



# - For the code below :
#   MIT License
#
#   Copyright (c) 2023 Anjok07 & aufr33 - Ultimate Vocal Remover (UVR 5)
#
# - https://github.com/Anjok07/ultimatevocalremovergui

MAX_SPEC = 'Max'
MIN_SPEC = 'Min'
AVERAGE  = 'Average'

def Make_Ensemble(algorithm, audio_input):

	if len(audio_input) == 1:  return audio_input[0]
	
	waves = []
	
	if algorithm == AVERAGE:

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
		output = waves / len(audio_input)
	else:
		specs = []
		
		for i in range(len(audio_input)):  
			waves.append(audio_input[i])
			
			# wave_to_spectrogram_no_mp
			spec = librosa.stft(audio_input[i], n_fft=6144, hop_length=1024)
			
			if spec.ndim == 1:  spec = np.asfortranarray([spec, spec])

			specs.append(spec)
		
		waves_shapes = [w.shape[1] for w in waves]
		target_shape = waves[waves_shapes.index(max(waves_shapes))]
		
		# spectrogram_to_wave_no_mp
		wave = librosa.istft(ensembling(algorithm, specs), n_fft=6144, hop_length=1024)
	
		if wave.ndim == 1:  wave = np.asfortranarray([wave, wave])

		output = to_shape(wave, target_shape.shape)

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
		elif MAX_SPEC == a:
			spec = np.where(np.abs(specs[i]) >= np.abs(spec), specs[i], spec)

	return spec

def to_shape(x, target_shape):
	padding_list = []
	for x_dim, target_dim in zip(x.shape, target_shape):
		pad_value = (target_dim - x_dim)
		pad_tuple = ((0, pad_value))
		padding_list.append(pad_tuple)
	
	return np.pad(x, tuple(padding_list), mode='constant')
