
#   MIT License - Copyright (c) 2023 Captain FLAM
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
		'model_instrum': "Instrum HQ 3",
		'model_vocals': "Kim Vocal 2",
	},
	'OPTIONS': {
		'bigshifts_MDX': 12,
		'overlap_MDX': 0.0,
#		'overlap_MDXv3': 8,
		'chunk_size': 500000,
		'use_SRS': True,
		'large_gpu': True,
	},
	'BONUS': {
		'TEST_MODE': False,
		'DEBUG': True,
		'GOD_MODE': False,
		'PREVIEWS': True,
	},
}

def Convert_to_Options(config):

	options = {}
	options['input']			= config['PATHS']['input']
	options['output']			= config['PATHS']['output']
	options['output_format']	= config['PROCESS']['output_format']
#	options['preset_genre']		= config['PROCESS']['preset_genre']
	options['model_instrum']	= config['PROCESS']['model_instrum']
	options['model_vocals']		= config['PROCESS']['model_vocals']
	options['bigshifts_MDX']	= int(config['OPTIONS']['bigshifts_MDX'])
	options['overlap_MDX']		= float(config['OPTIONS']['overlap_MDX'])
#	options['overlap_MDXv3']	= int(config['OPTIONS']['overlap_MDXv3'])
	options['chunk_size']		= int(config['OPTIONS']['chunk_size'])
	options['use_SRS']			= (config['OPTIONS']['use_SRS'].lower() == "true")
	options['large_gpu']		= (config['OPTIONS']['large_gpu'].lower() == "true")
	options['TEST_MODE']		= (config['BONUS']['TEST_MODE'].lower() == "true")
	options['DEBUG']			= (config['BONUS']['DEBUG'].lower() == "true")
	options['GOD_MODE']			= (config['BONUS']['GOD_MODE'].lower() == "true")
	options['PREVIEWS']			= (config['BONUS']['PREVIEWS'].lower() == "true")

	return options

def Load(Gdrive, isColab):

	global defaults
	file = os.path.join(Gdrive, "KaraFan", "Config_Colab.ini" if isColab else "Config_PC.ini")
	
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
	
	file = os.path.join(Gdrive, "KaraFan", "Config_Colab.ini" if isColab else "Config_PC.ini")

	with open(file, 'w') as config_file:
		config.write(config_file)
	