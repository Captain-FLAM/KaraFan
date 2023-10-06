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
		'preset': "Rock",
		'normalize': False,
		'vocals_1': "Kim Vocal 2",
		'vocals_2': "Voc FT",
		'vocals_3': "----",
		'vocals_4': "----",
		'REPAIR_MUSIC': "Maximum Mix",
		'instru_1': "Instrum HQ 3",
		'instru_2': "----",
		'filter_1': "Voc FT",
		'filter_2': "----",
		'filter_3': "----",
		'filter_4': "----"
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
		'large_gpu': True,
	},
}
Options = {
	'Output_format': [("FLAC - 24 bits", "FLAC"), ("MP3 - CBR 320 kbps", "MP3"), ("WAV - PCM 16 bits","PCM_16"), ("WAV - FLOAT 32 bits","FLOAT")],
	'Presets': ["🎵 Pop Rock", "🎵 Techno & SFX", "⏰ I'm a impatient !", "⭐ I want the best quality !", "🤩 * Special best SDR *"],
	'Speed': ['Fastest', 'Fast', 'Medium', 'Slow', 'Slowest'],
    'REPAIR_MUSIC': ["No !!", "Maximum Mix", "Average Mix"],
}
Presets = {
	"Rock": {
		'speed': "Medium",
		'vocals_1': "Kim Vocal 2",
		'vocals_2': "Voc FT",
		'vocals_3': "Kim Vocal 1",
		'vocals_4': "----",
		'REPAIR_MUSIC': "Maximum Mix",
		'instru_1': "Instrum HQ 3",
		'instru_2': "Instrum 3",
		'filter_1': "Kim Vocal 2",
		'filter_2': "Voc FT",
		'filter_3': "----",
		'filter_4': "----"
	},
	"Techno": {
		'speed': "Medium",
		'vocals_1': "Kim Vocal 2",
		'vocals_2': "Voc FT",
		'vocals_3': "Kim Vocal 1",
		'vocals_4': "----",
		'REPAIR_MUSIC': "Maximum Mix",
		'instru_1': "Instrum HQ 3",
		'instru_2': "Instrum 3",
		'filter_1': "Kim Vocal 1",
		'filter_2': "----",
		'filter_3': "----",
		'filter_4': "----"
	},
	"impatient": {
		'speed': "Fastest",
		'vocals_1': "Kim Vocal 2",
		'vocals_2': "Voc FT",
		'vocals_3': "----",
		'vocals_4': "----",
		'REPAIR_MUSIC': "No !!",
		'instru_1': "----",
		'instru_2': "----",
		'filter_1': "----",
		'filter_2': "----",
		'filter_3': "----",
		'filter_4': "----"
	},
	"quality": {
		'speed': "Slow",
		'vocals_1': "Kim Vocal 2",
		'vocals_2': "Voc FT",
		'vocals_3': "Kim Vocal 1",
		'vocals_4': "----",
		'REPAIR_MUSIC': "Maximum Mix",
		'instru_1': "Instrum HQ 3",
		'instru_2': "Instrum 3",
		'filter_1': "Kim Vocal 2",
		'filter_2': "Voc FT",
		'filter_3': "Kim Vocal 1",
		'filter_4': "----"
	},
	"SDR": {
		'speed': "Medium",
		'vocals_1': "Kim Vocal 2",
		'vocals_2': "Voc FT",
		'vocals_3': "Kim Vocal 1",
		'vocals_4': "----",
		'REPAIR_MUSIC': "No !!",
		'instru_1': "----",
		'instru_2': "----",
		'filter_1': "----",
		'filter_2': "----",
		'filter_3': "----",
		'filter_4': "----"
	}
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
	if  config['PROCESS']['preset'] not in Options['Presets']:
		config['PROCESS']['preset'] = Defaults['PROCESS']['preset']
	if  config['PROCESS']['REPAIR_MUSIC'] not in Options['REPAIR_MUSIC']:
		config['PROCESS']['REPAIR_MUSIC'] = Defaults['PROCESS']['REPAIR_MUSIC']
	if  config['OPTIONS']['speed'] not in Options['Speed']:
		config['OPTIONS']['speed'] = Defaults['OPTIONS']['speed']
		
	return config

def Save(Gdrive, isColab, config):
	
	file = os.path.join(Gdrive, "KaraFan_user", "Config_Colab.ini" if isColab else "Config_PC.ini")

	with open(file, 'w') as config_file:
		config.write(config_file)
	