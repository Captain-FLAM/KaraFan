
#   MIT License - Copyright (c) 2023 - Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan

import wx, wx.adv, traceback

def Report(exception, stack, object):
    
	text = ""
	for line in traceback.format_tb(stack):  text += line + "\n"
	text += exception

	# Open a modal dialog box to display the errors
	dialog = wx.Dialog(None, wx.ID_ANY, "KaraFan - Error", size = (800, 600), style = wx.DEFAULT_DIALOG_STYLE | wx.RESIZE_BORDER)
	dialog.Center()
	dialog.SetBackgroundColour(wx.Colour(255, 255, 255))
	dialog.SetIcon(wx.ArtProvider.GetIcon(wx.ART_ERROR, wx.ART_OTHER, (48, 48)))

	# Create a panel and a sizer
	panel = wx.Panel(dialog, wx.ID_ANY)
	sizer = wx.BoxSizer(wx.VERTICAL)
	sizer_links = wx.BoxSizer(wx.HORIZONTAL)

	# Title
	title = wx.StaticText(panel, wx.ID_ANY, "KaraFan - Error", style = wx.ALIGN_CENTER)
	title.SetFont(wx.Font(18, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD))
	sizer.Add(title, 0, wx.ALL | wx.EXPAND, 5)

	line = wx.StaticLine(panel, wx.ID_ANY, size = (800, 2), style = wx.LI_HORIZONTAL)
	sizer.Add(line, 0, wx.ALL | wx.EXPAND, 5)

	# Error message
	message = wx.StaticText(panel, wx.ID_ANY, exception, style = wx.ALIGN_CENTER)
	message.SetFont(wx.Font(14, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL))
	sizer.Add(message, 0, wx.ALL | wx.EXPAND, 5)

	line = wx.StaticLine(panel, wx.ID_ANY, size = (800, 2), style = wx.LI_HORIZONTAL)
	sizer.Add(line, 0, wx.ALL | wx.EXPAND, 5)
	
	# Stack trace (Selectable)
	stack = wx.TextCtrl(panel, wx.ID_ANY, text, style = wx.TE_MULTILINE | wx.TE_READONLY | wx.TE_DONTWRAP)
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
	button.Bind(wx.EVT_BUTTON, lambda event: dialog.Close())
	sizer.Add(button, 0, wx.ALL | wx.ALIGN_CENTER, 5)
	button.SetFocus()

	# Set the sizer and show the dialog
	panel.SetSizer(sizer)
	dialog.ShowModal()
	dialog.Destroy()

	# Check if object has a Close() method
	if object != None and hasattr(object, "Close"):  object.Close()
