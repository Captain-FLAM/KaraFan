#! python3.11

#   MIT License - Copyright (c) 2023 - Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan


# KaraFan works on your PC !!

import os, sys

Gdrive  = os.getcwd()
Project = os.path.join(Gdrive, "KaraFan")

# Are we in the Project directory ?
if not os.path.exists(Project):
	Gdrive  = os.path.dirname(Gdrive)  # Go to Parent directory
	Project = os.path.join(Gdrive, "KaraFan")
	os.chdir(Gdrive)

# Mandatory to import App modules
sys.path.insert(0, Project)

import App.main

App.main.Start({'Gdrive': Gdrive, 'Project': Project, 'isColab': False}, GUI = "wxpython")
