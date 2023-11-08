#! python3.11

#   MIT License - Copyright (c) 2023 - Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan


# KaraFan works on your PC !!

import os, sys, subprocess, platform, traceback

try:
	import wx, wx.adv
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

from Gui import Wx_main

try:
	app = wx.App(False)
	frame = Wx_main.KaraFanForm(None, {'Gdrive': Gdrive, 'Project': Project, 'isColab': False})
	frame.Show()
	app.MainLoop()

except Exception as e:
	# Open a modal dialog box to display the errors
	dialog = wx.App(False)
	frame  = wx.Frame(None, wx.ID_ANY, "KaraFan - Error", size = (800, 600))
	frame.Center()
	frame.SetBackgroundColour(wx.Colour(255, 255, 255))
	frame.SetIcon(wx.Icon(os.path.join(Project, "images", "KaraFan.ico")))

	# Create a panel and a sizer
	panel = wx.Panel(frame, wx.ID_ANY)
	sizer = wx.BoxSizer(wx.VERTICAL)
	sizer_links = wx.BoxSizer(wx.HORIZONTAL)

	# Title
	title = wx.StaticText(panel, wx.ID_ANY, "KaraFan - Error", style = wx.ALIGN_CENTER)
	title.SetFont(wx.Font(18, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
	sizer.Add(title, 0, wx.ALL | wx.EXPAND, 5)

	line = wx.StaticLine(panel, wx.ID_ANY, size = (800, 2), style = wx.LI_HORIZONTAL)
	sizer.Add(line, 0, wx.ALL | wx.EXPAND, 5)

	# Error message
	error = wx.StaticText(panel, wx.ID_ANY, f"{e.__class__.__name__} : {e.__str__()}", style = wx.ALIGN_CENTER)
	error.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
	sizer.Add(error, 0, wx.ALL | wx.EXPAND, 5)

	line = wx.StaticLine(panel, wx.ID_ANY, size = (800, 2), style = wx.LI_HORIZONTAL)
	sizer.Add(line, 0, wx.ALL | wx.EXPAND, 5)
	
	# Stack trace
	text = ""
	for line in traceback.format_tb(e.__traceback__):
		text += line + "\n"

	text += f"\n{e.__class__.__name__} : {e.__str__()}"

	stack = wx.StaticText(panel, wx.ID_ANY, text, style = wx.ALIGN_LEFT)
	stack.SetFont(wx.Font(12, wx.FONTFAMILY_TELETYPE, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
	sizer.Add(stack, 1, wx.ALL | wx.EXPAND, 5)

	line = wx.StaticLine(panel, wx.ID_ANY, size = (800, 2), style = wx.LI_HORIZONTAL)
	sizer.Add(line, 0, wx.ALL | wx.EXPAND, 5)

	# Bugs Reports
	contact = wx.StaticText(panel, wx.ID_ANY, "Report me these Bugs in private message on Discord, or open a new issue on GitHub !!", style = wx.ALIGN_CENTER)
	contact.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
	sizer.Add(contact, 0, wx.ALL | wx.EXPAND, 5)

	# Discord link
	link = wx.adv.HyperlinkCtrl(panel, wx.ID_ANY, "► Discord Invitation", "https://discord.gg/eXdEYwU2")
	link.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
	sizer_links.Add(link, 0, wx.RIGHT | wx.ALIGN_CENTER, 15)

	# Discord link
	link = wx.adv.HyperlinkCtrl(panel, wx.ID_ANY, "KaraFan channel on Discord", "https://discord.com/channels/708579735583588363/1162265179271200820")
	link.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
	sizer_links.Add(link, 0, wx.RIGHT | wx.ALIGN_CENTER, 15)

	# GitHub link
	link = wx.adv.HyperlinkCtrl(panel, wx.ID_ANY, "GitHub Issues", "https://github.com/Captain-FLAM/KaraFan/issues")
	link.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
	sizer_links.Add(link, 0, wx.RIGHT | wx.ALIGN_CENTER, 15)
	
	sizer.Add(sizer_links, 0, wx.ALL | wx.ALIGN_CENTER, 5)

	# Close
	button = wx.Button(panel, wx.ID_ANY, "OK")
	button.Bind(wx.EVT_BUTTON, lambda event: frame.Close())
	sizer.Add(button, 0, wx.ALL | wx.ALIGN_CENTER, 5)
	button.SetFocus()

	# Set the sizer and show the frame
	panel.SetSizer(sizer)
	frame.Show()
	dialog.MainLoop()
