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
		'vocals_1': "Kim Vocal 2",
		'vocals_2': "Voc FT",
#		'instru_1': "Instrum HQ 3",
#		'instru_2': "(None)",
#		'filter_1': "Kim Vocal 2",
#		'filter_2': "Voc FT",
#		'filter_3': "(None)",
#		'filter_4': "(None)"
	},
	'OPTIONS': {
		'shifts_vocals': 12,
#		'shifts_instru': 12,
#		'shifts_filter': 3,
#		'overlap_MDXv3': 8,
		'chunk_size': 500000,
	},
	'BONUS': {
		'KILL_on_END': False,
		'normalize': False,
		'DEBUG': False,
		'GOD_MODE': False,
		'PREVIEWS': False,
		'TEST_MODE': False,
		'large_gpu': True,
	},
}

def Convert_to_Options(config):

	options = {}
	options['input']			= config['PATHS']['input']
	options['output']			= config['PATHS']['output']
	options['output_format']	= config['PROCESS']['output_format']
#	options['preset_genre']		= config['PROCESS']['preset_genre']
	options['vocals_1']			= config['PROCESS']['vocals_1']
	options['vocals_2']			= config['PROCESS']['vocals_2']
#	options['instru_1']			= config['PROCESS']['instru_1']
#	options['instru_2']			= config['PROCESS']['instru_2']
#	options['filter_1']			= config['PROCESS']['filter_1']
#	options['filter_2']			= config['PROCESS']['filter_2']
#	options['filter_3']			= config['PROCESS']['filter_3']
#	options['filter_4']			= config['PROCESS']['filter_4']
	options['shifts_vocals']	= int(config['OPTIONS']['shifts_vocals'])
#	options['shifts_instru']	= int(config['OPTIONS']['shifts_instru'])
#	options['shifts_filter']	= int(config['OPTIONS']['shifts_filter'])
#	options['overlap_MDXv3']	= int(config['OPTIONS']['overlap_MDXv3'])
	options['chunk_size']		= int(config['OPTIONS']['chunk_size'])
	options['KILL_on_END']		= (config['BONUS']['KILL_on_END'].lower() == "true")
	options['normalize']		= (config['BONUS']['normalize'].lower() == "true")
	options['DEBUG']			= (config['BONUS']['DEBUG'].lower() == "true")
	options['GOD_MODE']			= (config['BONUS']['GOD_MODE'].lower() == "true")
	options['PREVIEWS']			= (config['BONUS']['PREVIEWS'].lower() == "true")
	options['TEST_MODE']		= (config['BONUS']['TEST_MODE'].lower() == "true")
	options['large_gpu']		= (config['BONUS']['large_gpu'].lower() == "true")

	return options

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
	