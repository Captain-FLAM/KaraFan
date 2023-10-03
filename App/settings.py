#!python3.10

#   MIT License - Copyright (c) 2023 - Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan

import os, configparser

# Default values
defaults = {
	'PATHS': {
		'input': "Music",
		'output': "Results",
	},
	'PROCESS': {
		'output_format': "FLAC",
#		'preset_genre': "Pop Rock",
		'normalize': False,
		'vocals_1': "Kim Vocal 2",
		'vocals_2': "Kim Vocal 1",
		'vocals_3': "Voc FT",
		'vocals_4': "(None)",
		'REPAIR_MUSIC': True,
		'instru_1': "Instrum HQ 3",
		'instru_2': "(None)",
#		'filter_1': "Kim Vocal 2",
#		'filter_2': "Voc FT",
#		'filter_3': "(None)",
#		'filter_4': "(None)"
	},
	'OPTIONS': {
		'quality': "Medium",
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

def Load(Gdrive, isColab):

	global defaults
	file = os.path.join(Gdrive, "KaraFan_user", "Config_Colab.ini" if isColab else "Config_PC.ini")
	
	config = configparser.ConfigParser()
	config.optionxform = lambda option: option  # To preserve case of Keys !!

	if os.path.isfile(file):
		config.read(file, encoding='utf-8')
		
		# Load default values if not present
		for section in defaults:
			if section not in config:
				config[section] = {}
			for key in defaults[section]:
				if key not in config[section]:
					config[section][key] = str(defaults[section][key])
	else:
		config.read_dict(defaults)
		Save(Gdrive, isColab, config)
	
	return config

def Save(Gdrive, isColab, config):
	
	file = os.path.join(Gdrive, "KaraFan_user", "Config_Colab.ini" if isColab else "Config_PC.ini")

	with open(file, 'w') as config_file:
		config.write(config_file)
	