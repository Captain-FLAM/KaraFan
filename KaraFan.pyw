#! python3.10

#   MIT License - Copyright (c) 2023 - Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan

# KaraFan works on your PC !!

import os, sys

Project = os.getcwd()  # Get the current path

# Are we in the Parent directory ?
if os.path.basename(Project) != "KaraFan":
	Project = os.path.join(Project, "KaraFan")
	
	if not os.path.exists(Project):
		print('ERROR : "KaraFan" folder not found !')
		os._exit(0)
	
	# os.chdir(Project)

# Mandatory to import App modules
sys.path.insert(0, Project)

import App.main

Gdrive = os.path.dirname(Project)  # Get Parent directory

params = {'Gdrive': Gdrive, 'Project': Project, 'isColab': False}

App.main.Start(params, GUI = "wxpython")
