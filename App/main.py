
import os, App.gui

def Start(params):

	# DEBUG : Auto-start processing on execution

	Auto_Start = 0

	song_output_path = os.path.join(params['Gdrive'], "Music", "song_017")

	# # Remove ALL files
	# for file in os.listdir(song_output_path):
	# 	if not file == "SDR_Results.txt":
	# 		os.remove(os.path.join(song_output_path, file))
	
	# Remove only Vocals files
	# for file in os.listdir(song_output_path):
	# 	if file.startswith("1"):
	# 		os.remove(os.path.join(song_output_path, file))

	# Remove only Music files
	# for file in os.listdir(song_output_path):
	# 	if file.startswith("2"):
	# 		os.remove(os.path.join(song_output_path, file))

	# # or just these ones
	# file = os.path.join(song_output_path, "1 - Vocal extract - (Voc FT).flac")
	# if os.path.isfile(file):  os.remove(file)
	# file = os.path.join(song_output_path, "1 - Vocal extract - (Kim Vocal 2).flac")
	# if os.path.isfile(file):  os.remove(file)

	App.gui.Run(params, Auto_Start)