
#   MIT License - Copyright (c) 2023 Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan

import os, configparser

def Load(Gdrive, isColab):

	file   = os.path.join(Gdrive, "KaraFan", "Config_Colab.ini" if isColab else "Config_PC.ini")
	
	config = configparser.ConfigParser()
	config.optionxform = lambda option: option  # To preserve case of Keys !!

	# Default values

	defaults = {
		'PATHS': {
			'input': "Music",
		},
		'PROCESS': {
			'output_format': "FLAC",
			'preset_genre': "Pop Rock",
		},
		'OPTIONS': {
			'bigshifts_MDX': 12,
			'overlap_MDX': 0.0,
#			'overlap_MDXv3': 8,
			'chunk_size': 500000,
			'use_SRS': True,
			'large_gpu': True,
		},
		'BONUS': {
			'DEBUG': True,
			'GOD_MODE': False,
			'PREVIEWS': True,
		},
	}

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
	
	file = os.path.join(Gdrive, "KaraFan", "Config_Colab.ini" if isColab else "Config_PC.ini")

	with open(file, 'w') as config_file:
		config.write(config_file)
	