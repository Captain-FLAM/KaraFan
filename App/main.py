
import os, Gui.notebook

def Start(params):
	
	Auto_Start = 0

	song_output_path = os.path.join(params['Gdrive'], "Music", "SDR_song_017")

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
	# file = os.path.join(song_output_path, "2 - Music extract - (Instrum HQ 3).flac")
	# if os.path.isfile(file):  os.remove(file)

	Gui.notebook.Run(params, Auto_Start)  # Auto-run processing on execution for DEBUG