#!/usr/bin/env python3.10

#   MIT License - Copyright (c) 2023 - Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan


#*************************************************
#****        DEBUG  ->  for DEVELOPERS        ****
#*************************************************

# Change to True if you're running on your PC (not in Visual Studio Code)
# ... and you don't want Auto-Updates !!

I_AM_A_DEVELOPER = False

#*************************************************

import os, App.setup, App.main

# We are on PC
Project = os.getcwd()  # Get the current path
Gdrive  = os.path.dirname(Project)  # Get parent directory

params = {'Gdrive': Gdrive, 'Project': Project, 'isColab': False, 'I_AM_A_DEVELOPER': I_AM_A_DEVELOPER}

# App.setup.Install(params)

App.main.Start(params, GUI = "wxpython")
