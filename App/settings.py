#!python3.10

#   MIT License - Copyright (c) 2023 - Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan

import os, configparser

# Default values
Defaults = {
	'PATHS': {
		'input': "Music",
		'output': "Results",
	},
	'PROCESS': {
		'output_format': "FLAC",
		'normalize': False,
		'vocals_1': "Kim Vocal 2",
		'vocals_2': "Voc FT",
		'vocals_3': "----",
		'vocals_4': "----",
		'REPAIR_MUSIC': "Max",
		'bleedings': "Soft",
		'instru_1': "Instrum HQ 3",
		'instru_2': "Instrum 3",
	},
	'OPTIONS': {
		'speed': "Medium",
#		'overlap_MDXv3': 8,
		'chunk_size': 500000,
	},
	'BONUS': {
		'KILL_on_END': False,
		'PREVIEWS': False,
		'DEBUG': False,
		'GOD_MODE': False,
		'TEST_MODE': False,
		'large_gpu': False,
	},
}
Options = {
	'Output_format': [("FLAC - 24 bits", "FLAC"), ("MP3 - CBR 320 kbps", "MP3"), ("WAV - PCM 16 bits","PCM_16"), ("WAV - FLOAT 32 bits","FLOAT")],
    'REPAIR_MUSIC': [("DON'T !!", "NO REPAIR"), ("Maximum Mix", 'Max'), ("Average Mix", 'Average')],
    'Bleedings': ["NO", "Soft", "Medium", "Hard"],
	'Speed': ['Fastest', 'Fast', 'Medium', 'Slow', 'Slowest'],
}
Help_Dico = {
	'input':		"- IF « Input » is a folder path, ALL audio files inside this folder will be separated by a Batch processing.<br>- Else, only the selected audio file will be processed.",
	'output':		"« Output folder » will be created based on the file\'s name without extension.<br>For example : if your audio input is named : « 01 - Bohemian Rhapsody<b>.MP3</b> »,<br>then output folder will be named : « 01 - Bohemian Rhapsody »",
	'format':		"Choose your prefered audio format to save audio files.",
	'normalize': 	"Normalize input audio files to avoid clipping and get better results.<br>Normally, <b>you do not have</b> to use this option !!<br>Only for weak or loud songs !",
	'MDX_vocal':	"Make an Ensemble of extractions with Vocals selected models.<br><br>Best combination : « <b>Kim Vocal 2</b> » and « <b>Voc FT</b> »",
	'repair':		"Repair music with <b>A.I</b> models.<br>Use it if you hear missing instruments, but ... <b>ALL models</b> will carry more or less <b>Vocal bleedings in Music Final</b> !!",
	'bleedings':	"Pass Music trough an <b>A.I</b> model to remove <b>Vocals Bleedings</b>.<br>If you want to keep <b>SFX</b> or hear missing instruments in music, don\'t use it !",
	'MDX_music':	"Make an Ensemble of instrumental extractions for repairing at the end of process.<br>Best combination : « <b>Instrum HQ 3</b> » and « <b>Instrum 3</b> » but ... <b>test</b> by yourself ! 😉<br>... You are warned : <b>ALL</b> instrumental models can carry <b>vocal bleedings</b> in final result !!",
	'speed':		"Fastest : extract in 1 pass with <b>NO</b> SRS and <b>NO</b> Denoise (<b>only</b> for Testing)<br>All others are multi-passes with <b>DENOISE</b> (the same option as in <b>UVR 5</b> 😉)<br>Slowest : is the best quality, but it will take hours to process !! 😝",
#	'MDX23c':		"MDX version 3 overlap. (default : 8)",
	'chunks':		"Chunk size for ONNX models. (default : 500,000)<br><br>Set lower to reduce GPU memory consumption OR <b>if you have GPU memory errors</b> !",
	'kill_end':		"On <b>Colab</b> : KaraFan will KILL your session at end of « Processongs », to save your credits !!<br>On <b>your Laptop</b> : KaraFan will KILL your GPU, to save battery (and hot-less) !!<br>On <b>your PC</b> : KaraFan will KILL your GPU, anyway ... maybe it helps ? Try it !!",
	'previews':		"Shows an audio player for each saved file. For impatients people ! 😉<br><br>(Preview first 60 seconds with quality of MP3 - VBR 192 kbps)",
	'debug':		"IF checked, it will save all intermediate audio files to compare in your <b>Audacity</b>.",
	'god_mode':		"Give you the GOD\'s POWER : each audio file is reloaded IF it was created before,<br>NO NEED to process it again and again !!<br>You\'ll be warned : You have to <b>delete MANUALLY</b> each file that you want to re-process !",
#	'large_gpu':	"It will load ALL models in GPU memory for faster processing of MULTIPLE audio files.<br>Requires more GB of free GPU memory.<br>Uncheck it if you have memory troubles.",
#	'reprocess':	"With <b>DEBUG</b> & <b>GOD MODE</b> activated : Available with <b>ONE file</b> at a time.<br>Automatic delete audio files of Stem that you want to re-process.<br>Vocals : <b>4_F</b> & <b>5_F</b> & <b>6</b>-Bleedings <b>/</b> Music : <b>same</b> + <b>2</b>-Music_extract & <b>3</b>-Audio_sub_Music",
}

def Load(Gdrive, isColab):

	file = os.path.join(Gdrive, "KaraFan_user", "Config_Colab.ini" if isColab else "Config_PC.ini")
	
	config = configparser.ConfigParser()
	config.optionxform = lambda option: option  # To preserve case of Keys !!

	if os.path.isfile(file):
		config.read(file, encoding='utf-8')
		
		# Load default values if not present
		for section in Defaults:
			if section not in config:
				config[section] = {}
			for key in Defaults[section]:
				if key not in config[section]:
					config[section][key] = str(Defaults[section][key])
	else:
		config.read_dict(Defaults)
		Save(Gdrive, isColab, config)
	
	if  config['PROCESS']['output_format'] not in [x[1] for x in Options['Output_format']]:
		config['PROCESS']['output_format'] = Defaults['PROCESS']['output_format']
	if  config['PROCESS']['REPAIR_MUSIC'] not in [x[1] for x in Options['REPAIR_MUSIC']]:
		config['PROCESS']['REPAIR_MUSIC'] = Defaults['PROCESS']['REPAIR_MUSIC']
	if  config['PROCESS']['bleedings'] not in Options['Bleedings']:
		config['PROCESS']['bleedings'] = Defaults['PROCESS']['bleedings']
	if  config['OPTIONS']['speed'] not in Options['Speed']:
		config['OPTIONS']['speed'] = Defaults['OPTIONS']['speed']
		
	return config

def Save(Gdrive, isColab, config):
	
	file = os.path.join(Gdrive, "KaraFan_user", "Config_Colab.ini" if isColab else "Config_PC.ini")

	with open(file, 'w') as config_file:
		config.write(config_file)
	