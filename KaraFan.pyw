#! python3.11

#   MIT License - Copyright (c) 2023 - Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan


# KaraFan works on your PC !!

import os, gc, sys, shutil, subprocess, platform

def on_rm_error(func, path, exc_info):
	# path contains the path of the file that couldn't be removed
	# let's just assume that it's read-only and unlink it.
	if platform.system() == 'Windows':
		subprocess.run(["attrib", "-r", path], text=True, capture_output=True, check=True)
	else:
		os.chmod(path, 0o777)  # Linux & Mac
	os.remove(path)


Gdrive  = os.getcwd()
Project = os.path.join(Gdrive, "KaraFan")

# Are we in the Project directory ?
if not os.path.exists(Project):
	Gdrive  = os.path.dirname(Gdrive)  # Go to Parent directory
	Project = os.path.join(Gdrive, "KaraFan")
	os.chdir(Gdrive)

# Temporary fix for old KaraFan < 5.1
old_version = os.path.join(Gdrive, "KaraFan-master")
if os.path.exists(old_version) and os.path.isdir(old_version):  shutil.rmtree(Project, onerror = on_rm_error)

# Temporary fix for old KaraFan < 5.1.3
old_version = os.path.join(Gdrive, "KaraFan.py")
if os.path.exists(old_version) and os.path.isfile(old_version):  os.remove(old_version)


try:
	import wx
except:
	if platform.system() == "Windows":
		subprocess.run(["cmd", "/c", "@cls & @echo. & @echo wxPython is not installed ! & echo. & echo Run 'Setup.py' first !! & echo. & pause"], shell = True)
	else:
		subprocess.run(["bash", "-c", 'clear ; echo "wxPython is not installed !" ; echo ; echo "Run \'Setup.py\' first !!" ; echo ; read -n 1 -s -r -p "Press any key to continue..."'], shell = True)

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
	gc.collect()
	
except Exception as e:
	Gui.wx_Error.Report(f"{e.__class__.__name__} : {e}", sys.exc_info()[2], frame)
