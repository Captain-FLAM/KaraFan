
#   MIT License - Copyright (c) 2023 - Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan

import os

def Start(params, GUI = "ipywidgets"):
	
	#*************************************************
	#****        DEBUG  ->  for DEVELOPERS        ****
	#*************************************************

	# Auto-run processing on execution for quick DEBUGGING
	
	params['Auto_Start'] = False

	song_output_path = os.path.join(params['Gdrive'], "Music", "SDR_song_017")

	# # Remove ALL files
	# for file in os.listdir(song_output_path):
	# 	if not file == "SDR_Results.txt":
	# 		os.remove(os.path.join(song_output_path, file))
	
	# Remove only Music extract
	# for file in os.listdir(song_output_path):
	# 	if file.startswith("1"):
	# 		os.remove(os.path.join(song_output_path, file))

	# Remove only Vocals extract
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

	#*************************************************
	
	if GUI == "ipywidgets":
		from Gui import Notebook
		Notebook.Run(params)
	
	elif GUI == "wxpython":
		import wx
		from Gui import Wx_main
		app = wx.App(False)
		frame = Wx_main.KaraFanForm(None, params)
		frame.Show()
		app.MainLoop()
