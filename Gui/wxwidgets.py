#!python3.10

#   MIT License - Copyright (c) 2023 - Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan

import os
import csv
import glob
import wx
import wx.html as html

from App import settings, inference, sys_info
from Gui import progress

Running = False

def Run(params, Auto_Start):

	Gdrive = params['Gdrive']
	Project = params['Project']
	isColab = params['isColab']
	DEV_MODE = params['I_AM_A_DEVELOPER']

	width = 670
	height = 600

	# Set the font size when running on your PC
	font = 14
	font_small = 13

	# Set the font size when running on Colab
	if isColab:
		font = 15
		font_small = 13

	font_input = wx.Font(font, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL)
	panel_layout = wx.BoxSizer(wx.VERTICAL)
	checkbox_layout = wx.BoxSizer(wx.VERTICAL)
	max_width = width - 18  # = border + Left and Right "panel_layout" padding
	console_max_height = height - 20

	# Get local version
	with open(os.path.join(Project, "App", "__init__.py"), "r") as version_file:
		Version = version_file.readline().replace("# Version", "").strip()

	# Get config values
	config = settings.Load(Gdrive, isColab)

	vocals = ["----"]
	instru = ["----"]
	filters = ["----"]
	
	with open(os.path.join(Project, "App", "Models_DATA.csv")) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			if row['Use'] == "x":
				if row['Stem'] == "Instrumental":
					instru.append(row['Name'])
				elif row['Stem'] == "Vocals":
					vocals.append(row['Name'])
