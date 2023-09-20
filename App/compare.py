#!python3.10

#   Copyright (c) 2021 - Roman Solovyev (ZFTurbo), IPPM RAS
#   Copyright (c) 2023 Captain FLAM - Heavily modified to use with KaraFan
#
#   https://github.com/ZFTurbo/Audio-separation-models-checker

import os, glob
import librosa, soundfile as sf
import numpy as np

"""
SDR - Source to distortion ratio   
SIR - Source to inferences ratio
SNR - Source to noise ration
SAR - Source to artifacts ratio
"""

def calculate(references, estimates):
	# compute SDR for one song
	delta = 1e-7  # avoid numerical errors
	num = np.sum(np.square(references), axis=(1, 2))
	den = np.sum(np.square(references - estimates), axis=(1, 2))
	num += delta
	den += delta
	return 10 * np.log10(num / den)

def SDR(song_output_path, Gdrive):

	song_name = os.path.basename(song_output_path)
	
	MultiSong_path = os.path.join(Gdrive, "KaraFan_user", "Multi_Song", song_name)

	if not os.path.exists(MultiSong_path):
		return

	Extracted_files = glob.glob(os.path.join(song_output_path, '*.flac'))

	if len(Extracted_files) == 0:  print('Check output folder. Cant find any files !');  return

	scores = {"instrum": [], "vocals": []}  # The stems we want to process

	for extract in Extracted_files:

		# Skip Bleedings
		if "Bleedings" in extract: continue

		# If the file contains "Vocals", check the vocals
		if "Vocal" in extract:		type = "vocals"
		elif "Music" in extract:	type = "instrum"
		else:
			continue  # Skip Others

		reference, _ = sf.read(os.path.join(MultiSong_path, type + '.flac'))
		estimate, _  = sf.read(extract)

		references = np.expand_dims(reference, axis=0)
		estimates  = np.expand_dims(estimate, axis=0)

		if estimates.shape != references.shape:
			print('Warning: Different length of FLAC files : {} != {}. Skip it !'.format(estimates.shape, references.shape))
			continue

		song_score = calculate(references, estimates)[0]

		print('► ' + os.path.splitext(os.path.basename(extract))[0] + ' - SDR : <b>{:.6f}</b>'.format(song_score))

		scores[type].append(song_score)

	for type in scores:
		if len(scores[type]) > 0:
			print('Average SDR {} : <b>{:.6f}</b>'.format(type, np.array(scores[type]).mean()))
		else:
			print('Average SDR {} : ---'.format(type))



#   MIT License - Copyright (c) 2023 Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan

def Spectrograms(audio_file1, audio_file2):
	
	audio1, _ = librosa.load(audio_file1, sr=None, mono=False)
	audio2, _ = librosa.load(audio_file2, sr=None, mono=False)

	# Convertir les signaux audio en spectrogrammes
	spec1 = librosa.stft(audio1, n_fft=4096, hop_length=1024)
	spec2 = librosa.stft(audio2, n_fft=4096, hop_length=1024)

	# Calculer la distance euclidienne entre les spectrogrammes
	distance = np.linalg.norm(spec1 - spec2)

	# Normaliser la distance pour obtenir une mesure de similarité (plus proche de zéro est meilleur)
	similarity = 1 / (1 + distance)

	return f"{similarity * 100:.6f} %"
