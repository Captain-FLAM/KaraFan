#!python3.10

#   Copyright (c) 2021 - Roman Solovyev (ZFTurbo), IPPM RAS
#   Copyright (c) 2023 Captain FLAM - Heavily modified to use with KaraFan
#
#   https://github.com/ZFTurbo/Audio-separation-models-checker

import os, glob, datetime, librosa, soundfile as sf, numpy as np

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

def SDR(song_output_path, output_format, Gdrive):

	song_name = os.path.basename(song_output_path).replace("SDR_", "")
	
	MultiSong_path = os.path.join(Gdrive, "KaraFan_user", "Multi_Song", song_name)

	if not os.path.exists(MultiSong_path):
		return

	match output_format:
		case 'PCM_16':	ext = '.wav'
		case 'FLOAT':	ext = '.wav'
		case "FLAC":	ext = '.flac'
		case 'MP3':		ext = '.mp3'

	Extracted_files = glob.glob(os.path.join(song_output_path, '*' + ext))

	if len(Extracted_files) == 0:  print('Check output folder. Cant find any files !');  return

	Scores  = {"instrum": [], "vocals": []}  # The stems we want to process
	Results = ""

	for extract in Extracted_files:

		file_name = os.path.basename(extract)

		# Skip Bleedings
		if "Bleedings" in file_name:  continue

		if "Vocal" in file_name:	type = "vocals"
		elif "Music" in file_name:	type = "instrum"
		else:
			continue  # Skip Others

		reference, _ = sf.read(os.path.join(MultiSong_path, type + '.flac'))
		estimate, _  = sf.read(extract)

		references = np.expand_dims(reference, axis=0)
		estimates  = np.expand_dims(estimate, axis=0)

		if estimates.shape != references.shape:
			print('Warning: Different length of files : {} != {}. Skip it !'.format(estimates.shape, references.shape))
			continue

		song_score = calculate(references, estimates)[0]

		file_name = os.path.splitext(os.path.basename(extract))[0]
		pad = 40 - len(file_name)

		Results += file_name + (" " * pad) + 'SDR : {:9.6f}'.format(song_score) + "\n"

		print('• ' + file_name + ("&nbsp;" * pad) + 'SDR : <b>{:9.6f}</b>'.format(song_score))

		Scores[type].append(song_score)

	# TODO : Use for batch SDR with multiple songs
	# for type in Scores:
	# 	if len(Scores[type]) > 0:
	# 		print('Average SDR {} : <b>{:9.6f}</b>'.format(type, np.array(Scores[type]).mean()))
	# 	else:
	# 		print('Average SDR {} : ---'.format(type))

	# Write results on disk
	if Results != "":
		with open(os.path.join(song_output_path, "SDR_Results.txt"), 'a', encoding='utf-8') as file:
			file.write(f"\n► {datetime.datetime.now().strftime('%Y-%m-%d ~ %H:%M:%S')} - {song_name}\n\n")
			file.write(Results)


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
