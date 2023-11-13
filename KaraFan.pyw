#! python3.11

#   MIT License - Copyright (c) 2023 - Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan


# KaraFan works on your PC !!

import os, sys, subprocess, platform

try:
	import wx
except:
	if platform.system() == "Windows":
		subprocess.run(["cmd", "/c", "@cls & @echo. & @echo wxPython is not installed ! & echo. & echo Run 'Setup.py' first !! & echo. & pause"], shell = True)
	else:
		subprocess.run(["bash", "-c", 'clear ; echo "wxPython is not installed !" ; echo ; echo "Run \'Setup.py\' first !!" ; echo ; read -n 1 -s -r -p "Press any key to continue..."'], shell = True)

Gdrive  = os.getcwd()
Project = os.path.join(Gdrive, "KaraFan")

# Are we in the Project directory ?
if not os.path.exists(Project):
	Gdrive  = os.path.dirname(Gdrive)  # Go to Parent directory
	Project = os.path.join(Gdrive, "KaraFan")
	os.chdir(Gdrive)

# Mandatory to import Gui & App modules
sys.path.insert(0, Project)

import Gui.wx_Main, Gui.wx_Error

# Global ERROR handler
def error_handler(exception, value, traceback):
	if exception.__name__ == "SystemError":
		Gui.wx_Error.Report(f"{exception.__name__} : {value}", traceback, None)
	else:
		Gui.wx_Error.Report(f"{exception.__name__} : {value}", traceback, value.obj)

sys.excepthook = error_handler

frame = None
try:
	app = wx.App(False, useBestVisual=True)
	frame = Gui.wx_Main.KaraFanForm(None, {'Gdrive': Gdrive, 'Project': Project, 'isColab': False})
	frame.Show()
	app.MainLoop()
	
except Exception as e:
	Gui.wx_Error.Report(f"{e.__class__.__name__} : {e}", sys.exc_info()[2], frame)
