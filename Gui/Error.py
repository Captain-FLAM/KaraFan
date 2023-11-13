
#   MIT License - Copyright (c) 2023 - Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan

import traceback

def Report(exception, stack):
    
	text = ""
	for line in traceback.format_tb(stack):
		line = line.replace('\n', '<br>')
		line = line.replace(' ', '&nbsp;')
		text += line + '<br>'
	
	text += exception

	# Print the message in the console
	print('<hr>')
	print('<div style="color:#ff0000;"><b>KaraFan - Error !!</b></div>')
	print('<hr>')
	print('<pre style="white-space:nowrap;overflow:auto;" nowrap="nowrap">' + text + '</pre>')
	print('<hr>')
	print('► Report me <b>this Bug above</b> in private message on Discord :')
	print('<a href="https://discord.gg/eXdEYwU2">Discord Invitation</a> / <a href="https://discord.com/channels/708579735583588363/1162265179271200820">KaraFan channel on Discord</a>')
	print()
	print('► or open a new issue on GitHub : <a href="https://github.com/Captain-FLAM/KaraFan/issues">GitHub Issues</a>')
	print('<hr>')
	print()
