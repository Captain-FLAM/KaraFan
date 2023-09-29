#!python3.10

#   MIT License
#
#   Copyright (c) 2023 - Captain FLAM
#   Copyright (c) 2021 - Roman Solovyev (ZFTurbo)
#
#   https://github.com/Captain-FLAM/KaraFan
#   https://github.com/ZFTurbo/Audio-separation-models-checker

import os, glob, datetime, librosa, soundfile as sf, numpy as np

"""
SDR - Source to distortion ratio   
SIR - Source to inferences ratio
SNR - Source to noise ration
SAR - Source to artifacts ratio
"""

# compute SDR for one song
def calculate(reference, estimate):
	references = np.expand_dims(reference, axis=0)
	estimates  = np.expand_dims(estimate, axis=0)

	if estimates.shape != references.shape:
		print('Warning: Different length of files : {} != {}. Skip it !'.format(estimates.shape, references.shape))
		return [None]

	delta = 1e-7  # avoid numerical errors
	num = np.sum(np.square(references), axis=(1, 2))
	den = np.sum(np.square(references - estimates), axis=(1, 2))
	num += delta
	den += delta
	return 10 * np.log10(num / den)

def SDR(song_output_path, output_format, Gdrive):

	# The "song_output_path" contains the NAME of the song to compare within the "Gdrive > KaraFan_user > Multi-Song" folder

	song_name		= os.path.basename(song_output_path)
	MultiSong_path	= os.path.join(Gdrive, "KaraFan_user", "Multi_Song", "Stems")

	if not os.path.exists(MultiSong_path):  return

	match output_format:
		case 'PCM_16':	ext = 'wav'
		case 'FLOAT':	ext = 'wav'
		case "FLAC":	ext = 'flac'
		case 'MP3':		ext = 'mp3'

	Extracted_files = glob.glob(os.path.join(song_output_path, '*.' + ext))

	if len(Extracted_files) == 0:  print('Check output folder. Cant find any files !');  return

	 # The stems we want to process
	Scores			= {"instrum": [], "vocals": []}
	References		= {"instrum": None, "vocals": None}

	Results = ""
	References["instrum"], _ = sf.read(os.path.join(MultiSong_path, song_name[-3:] + '_instrum.flac'))  # Get only the number of the song
	References["vocals"], _  = sf.read(os.path.join(MultiSong_path, song_name[-3:] + '_vocals.flac'))

	Extracted_files = sorted(Extracted_files)

	for extract in Extracted_files:

		file_name = os.path.basename(extract)
		file_name = file_name.replace(os.path.splitext(file_name)[1], "")  # Remove extension

		if "Bleedings" in file_name:  continue  # Skip Bleedings

		if "Vocal" in file_name:	type = "vocals"
		elif "Music" in file_name:	type = "instrum"
		else:						type = "others"

		if type == "others":  continue  # Skip Others

		estimate, _	= sf.read(extract)
		song_score	= calculate(References[type], estimate)[0]

		if not song_score is None:
			pad = 40 - len(file_name)

			print("• " + file_name + ("&nbsp;" * pad) + 'SDR : <b>{:9.6f}</b>'.format(song_score))
			Results += file_name + (" " * pad) + 'SDR : {:9.6f}'.format(song_score) + "\n"

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

def SDR_Volumes(type, audio, song_output_path, Gdrive):

	# The "song_output_path" contains the NAME of the song to compare within the "Gdrive > KaraFan_user > Multi-Song" folder

	song_name		= os.path.basename(song_output_path)
	MultiSong_path	= os.path.join(Gdrive, "KaraFan_user", "Multi_Song", "Stems")

	if not os.path.exists(MultiSong_path):  return

	if type == "Vocal":		type = "vocals"
	elif type == "Music":	type = "instrum"
	else:					return

	Scores = [];  Volumes = []
	audio  = audio.T
	reference, _ = sf.read(os.path.join(MultiSong_path, song_name[-3:] + '_' + type + '.flac'))
	
	for i in range(0, 95, 1):
		volume = round(0.94 + (i * 0.001), 3)
		song_score = calculate(reference, audio * volume)[0]

		if not song_score is None:
			Scores.append(song_score)
			Volumes.append(volume)

	# Show Best Volume Compensation
	if len(Scores) > 0:
		SDR		= max(Scores)
		volume	= Volumes[Scores.index(SDR)]

		print("Best Volume Compensation : {} - ({} to {})- <b>{:9.6f}</b>".format(volume, min(Volumes), max(Volumes), SDR))
		
def Spectrograms(audio_file1, audio_file2):
	
	audio1, _ = librosa.load(audio_file1, sr=None, mono=False)
	audio2, _ = librosa.load(audio_file2, sr=None, mono=False)

	# Convertir les signaux audio en spectrogrammes
	spec1 = librosa.stft(audio1, n_fft=4096, hop_length=1024)
	spec2 = librosa.stft(audio2, n_fft=4096, hop_length=1024)

	# Calculer la distance euclidienne entre les spectrogrammes
	distance = np.linalg.norm(spec2 - spec1)

	# Normaliser la distance pour obtenir une mesure de similarité (plus proche de zéro est meilleur)
	similarity = 1 / (1 + distance)

	return f"{similarity * 100:.6f} %"
