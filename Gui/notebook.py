#!python3.10

#   MIT License - Copyright (c) 2023 - Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan

import os, csv, glob

import ipywidgets as widgets
from IPython.display import display, HTML

Running = False

def Run(params, Auto_Start):

	import App.settings, App.inference, App.sys_info, Gui.progress

	Gdrive = params['Gdrive']
	Project = params['Project']
	isColab = params['isColab']
	DEV_MODE = params['I_AM_A_DEVELOPER']

	width  = '670px'
	height = '600px'

	# Set the font size when running on your PC
	font = '14px'

	if isColab:  font = '15px';

	font_input = {'font_size': font}
	panel_layout = {'height': height, 'max_height': height, 'margin':'8px'}
	checkbox_layout = {'width': '30px', 'max_width': '30px' }
	max_width = str(int(width.replace('px','')) - 18) + 'px'  # = border + Left and Right "panel_layout" padding
	console_max_height = str(int(height.replace('px','')) - 20) + 'px'

	# This CSS is the style for HTML elements for BOTH : PC and Colab
	# BUG on Colab: "style={'font_size':'16px'}" as widgets param doesn't work !!
	# I've fixed it with this Trick --> #output_body > element { font-size: 16px; }
	display(HTML(
'<style>\
#output-body input, #output-body button, #output-body select, #output-body select > option, #output-body .widget-readout { font-size: '+ font +' }\
#output-body .lm-TabBar-tabLabel, .lm-TabBar-tabLabel { font-size: 16px; padding-top: 5px }\
#output-body .progress-bar-success, .progress-bar-success { background-color: lightblue}\
.option-label { font-size: '+ font +'; width: 135px }\
.short-label { font-size: '+ font +' }\
#HELP { font-size: '+ font +'; background-color: #ffffd2; border: solid 1px #333; width: 100%; height: 63px; line-height: 1.2 }\
#HELP > div { margin: 5px 10px }\
.console { font: normal '+ font +' monospace; line-height: 1.6 }\
.player { margin-bottom: 5px }\
.player > div { min-width: 200px; display: inline-block; font: normal '+ font +' monospace }\
.player > audio { vertical-align: middle }\
.SDR { display: inline-block; line-height: 1 }\
</style>'))
	
	def Label(text, index, short = False):
		return widgets.HTML('<div class="'+ ('short' if short else 'option') + f'-label" onmouseenter="show_help(\'{index}\')">{text}</div>')
	
	# Get local version
	with open(os.path.join(Project, "App", "__init__.py"), "r") as version_file:
		Version = version_file.readline().replace("# Version", "").strip()

	# Get config values
	config = App.settings.Load(Gdrive, isColab)

	# Fill Models dropdowns
	vocals = ["----"]; instru = ["----"]
	with open(os.path.join(Project, "Data", "Models.csv")) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			if row['Use'] == "x":
				# New fine-tuned MDX23C (less vocal bleedings in music)
				if row['Name'] == "MDX23C 8K FFT - v2" and not os.path.isfile(os.path.join(Gdrive, "KaraFan_user", "Models", "MDX23C-8KFFT-InstVoc_HQ_2.ckpt")):
					continue
				# ignore "Other" stems
				if row['Stem'] == "BOTH":			instru.append(row['Name']);  vocals.append(row['Name'])
				elif row['Stem'] == "Instrumental":	instru.append(row['Name'])
				elif row['Stem'] == "Vocals":		vocals.append(row['Name'])
	
	# KaraFan Title
	display(HTML('<div style="font-size: 24px; font-weight: bold; margin: 15px 0">KaraFan - version '+ Version +'</div>'))
	
	# TABS
	titles = ["☢️ Settings", "♾️ Progress", "❓ System Info"]

	# TAB 1
	separator		= widgets.HTML('<div style="border-bottom: dashed 1px #000; margin: 5px 0 5px 0; width: 100%">')
	# AUDIO
	input_path		= widgets.Text(config['AUDIO']['input'], continuous_update=True, layout = {'width':'310px'}, style=font_input)
	output_path		= widgets.Text(config['AUDIO']['output'], continuous_update=True, layout = {'width':'310px'}, style=font_input)
	output_format	= widgets.Dropdown(value = config['AUDIO']['output_format'], options = App.settings.Options['Output_format'], layout = {'width':'150px'}, style=font_input)
	normalize		= widgets.Checkbox((config['AUDIO']['normalize'].lower() == "true"), indent=False, style=font_input, layout=checkbox_layout)
	silent			= widgets.Dropdown(value = config['AUDIO']['silent'], options = App.settings.Options['Silent'], layout = {'width':'100px'}, style=font_input)
	# PROCESS
	music_1			= widgets.Dropdown(options = instru, layout = {'width':'200px'}, style=font_input)
	music_2			= widgets.Dropdown(options = instru, layout = {'width':'200px'}, style=font_input)
	vocal_1			= widgets.Dropdown(options = vocals, layout = {'width':'200px'}, style=font_input)
	vocal_2			= widgets.Dropdown(options = vocals, layout = {'width':'200px'}, style=font_input)
	bleed_1			= widgets.Dropdown(options = instru, layout = {'width':'200px'}, style=font_input)
	bleed_2			= widgets.Dropdown(options = instru, layout = {'width':'200px'}, style=font_input)
	bleed_3			= widgets.Dropdown(options = vocals, layout = {'width':'200px'}, style=font_input)
	bleed_4			= widgets.Dropdown(options = vocals, layout = {'width':'200px'}, style=font_input)
	bleed_5			= widgets.Dropdown(options = instru, layout = {'width':'200px'}, style=font_input)
	bleed_6			= widgets.Dropdown(options = instru, layout = {'width':'200px'}, style=font_input)
	# OPTIONS
	speed			= widgets.SelectionSlider(value = config['OPTIONS']['speed'], options = App.settings.Options['Speed'], readout=True, style=font_input) 
	chunk_size		= widgets.IntSlider(int(config['OPTIONS']['chunk_size']), min=100000, max=1000000, step=100000, readout_format = ',d', style=font_input)
	# BONUS
	KILL_on_END		= widgets.Checkbox((config['BONUS']['KILL_on_END'].lower() == "true"), indent=False, style=font_input, layout=checkbox_layout)
	PREVIEWS		= widgets.Checkbox((config['BONUS']['PREVIEWS'].lower() == "true"), indent=False, style=font_input, layout=checkbox_layout)
	DEBUG			= widgets.Checkbox((config['BONUS']['DEBUG'].lower() == "true"), indent=False, continuous_update=True, style=font_input, layout=checkbox_layout)
	GOD_MODE		= widgets.Checkbox((config['BONUS']['GOD_MODE'].lower() == "true"), indent=False, continuous_update=True, style=font_input, layout=checkbox_layout)
	# TODO : Large GPU -> Do multiple Pass with steps with 3 models max for each Song
	# large_gpu		= widgets.Checkbox((config['BONUS']['large_gpu'].lower() == "true"), indent=False, style=font_input, layout=checkbox_layout)
	Btn_Del_Vocals	= widgets.Button(description='Vocals', button_style='danger', layout={'width':'80px', 'margin':'0 20px 0 0'})
	Btn_Del_Music	= widgets.Button(description='Music',  button_style='danger', layout={'width':'80px', 'margin':'0 20px 0 0'})
	# +
	Btn_Preset_1	= widgets.Button(description='1️⃣', tooltip="Preset 1", layout={'width':'72px', 'height':'27px', 'margin':'10px 0 0 5px'}, style={'font_size': '22px', 'button_color':'#fff'})
	Btn_Preset_2	= widgets.Button(description='2️⃣', tooltip="Preset 2", layout={'width':'72px', 'height':'27px', 'margin':'10px 0 0 5px'}, style={'font_size': '22px', 'button_color':'#fff'})
	Btn_Preset_3	= widgets.Button(description='3️⃣', tooltip="Preset 3", layout={'width':'72px', 'height':'27px', 'margin':'10px 0 0 5px'}, style={'font_size': '22px', 'button_color':'#fff'})
	Btn_Preset_4	= widgets.Button(description='4️⃣', tooltip="Preset 4", layout={'width':'72px', 'height':'27px', 'margin':'10px 0 0 5px'}, style={'font_size': '22px', 'button_color':'#fff'})
	Btn_Start		= widgets.Button(description='Start', button_style='primary', layout={'width':'200px', 'margin':'8px 0 12px 0'})
	HELP			= widgets.HTML('<div id="HELP"></div>')

	# TAB 2
	CONSOLE			= widgets.Output(layout = {'max_width': max_width, 'height': console_max_height, 'max_height': console_max_height, 'overflow':'scroll'})
	Progress_Bar	= widgets.IntProgress(value=0, min=0, max=10, orientation='horizontal', bar_style='info', layout={'width':'370px', 'height':'25px'})  # style={'bar_color': 'maroon'})
	Progress_Text	= widgets.HTML(layout={'width':'300px', 'margin':'0 0 0 15px'})
	# +
	Progress		= Gui.progress.Bar(Progress_Bar, Progress_Text)  # Class for Progress Bar

	# TAB 3
	sys_info		= widgets.HTML()
	Btn_SysInfo		= widgets.Button(description='Get System informations', button_style='primary', layout={'width':'200px'})

	# Create TABS
	tab = widgets.Tab(layout = {'max_width': width, 'margin':'0'})
	tab.children = [
		widgets.VBox(
			layout = panel_layout,
			children = [
				widgets.VBox([
					widgets.HBox([ Label("Input X file or PATH", 'input'), input_path, Label("&nbsp; Normalize input &nbsp;", 'normalize', short=True), normalize ]),
						
					widgets.HBox([ Label("Output PATH", 'output'), output_path ]),
					widgets.HBox([
						Label("Output Format", 'format'), output_format,
						Label("&nbsp; Silent &nbsp;", 'silent', short=True), silent
					]),
				]),
				separator,
				widgets.VBox([
					widgets.HBox([ Label("Filter Music", 'MDX_music'),		music_1, music_2, widgets.HTML('<span style="font-size:18px">&nbsp; 🎵</span>') ]),
					widgets.HBox([ Label("Extract Vocals", 'MDX_vocal'),	vocal_1, vocal_2, widgets.HTML('<span style="font-size:18px">&nbsp; 💋</span>') ]),
					widgets.HBox([ Label("Music Bleedings", 'MDX_bleed_1'), bleed_1, bleed_2, widgets.HTML('<span style="font-size:18px">&nbsp; 🎵</span>') ]),
				]),
				separator,
				widgets.VBox([
					widgets.HBox([ Label("Vocal Bleedings", 'MDX_bleed_2'), bleed_3, bleed_4, widgets.HTML('<span style="font-size:18px">&nbsp; 💋</span>') ]),
					widgets.HBox([ Label("Remove Music", 'MDX_bleed_3'),	bleed_5, bleed_6, widgets.HTML('<span style="font-size:18px">&nbsp; 🎵</span>') ]),
				]),
				separator,
				widgets.VBox([
					widgets.HBox([ Label("Speed", 'speed'), speed ]),
					widgets.HBox([ Label("Chunk Size", 'chunks'), chunk_size ]),
				]),
				separator,
				widgets.VBox([
					widgets.HBox([ Label("This is the END ...", 'kill_end'), KILL_on_END, Label("Show Previews", 'previews'), PREVIEWS ]),
					widgets.HBox([ Label("DEBUG Mode", 'debug'), DEBUG, Label("GOD Mode", 'god_mode'), GOD_MODE ]),
					# TODO : Large GPU -> Do multiple Pass with steps with 2 models max for each Song
#					, Label('Large GPU', 'large_gpu'), large_gpu ]),
#					widgets.HBox([ Label("RE-Process ▶️▶️", 'reprocess'), Btn_Del_Vocals, Btn_Del_Music ], layout={'margin':'15px 0 0 0'}),
				]),
				separator,
				widgets.HBox([
					widgets.HBox([Btn_Preset_1, Btn_Preset_2, Btn_Preset_3, Btn_Preset_4]),
					widgets.HBox([Btn_Start], layout={'width':'100%', 'justify_content':'center'}),
				]),
				HELP
			]),
		widgets.VBox(
			layout = panel_layout,
			children = [
				CONSOLE,
				widgets.HBox([ Progress_Bar, Progress_Text ], layout={'width':'100%', 'height': '30px', 'margin':'10px 0 0 0'}),
			]),
		widgets.VBox(
			layout = panel_layout,
			children = [
				widgets.VBox([Btn_SysInfo]),
				sys_info
			]),
	]

	# Bug in VS Code : titles NEEDS to be set AFTER children
	tab.titles = titles
	tab.selected_index = 0
	display(tab)

	#*********************
	#**  Buttons Click  **
	#*********************
	
	def on_Start_clicked(b):
		global Running

		HELP.value = '<div id="HELP"></div>'  # Clear HELP
		msg = ""
		if input_path.value == "":
			msg += "Input is required !<br>"
		else:
			path = os.path.join(Gdrive, input_path.value)
			if not os.path.isfile(path) and not os.path.isdir(path):
				msg += "Your Input is not a valid file or folder !<br>You MUST set it to an existing audio file or folder path.<br>"
		
		if output_path.value == "":
			msg += "Output is required !<br>"
		else:
			path = os.path.join(Gdrive, output_path.value)
			if not os.path.isdir(path):
				msg += "Your Output is not a valid folder !<br>You MUST set it to an existing folder path.<br>"
		
		if vocal_1.value == "----" and vocal_2.value == "----":
			msg += "You HAVE TO select at least one model for Vocals !<br>"
		
		if msg != "":
			msg = "ERROR !!<br>"+ msg
			HELP.value = '<div id="HELP"><div style="color: #f00">'+ msg +'</div></div>'
			return
		
		input_path.value = os.path.normpath(input_path.value)
		output_path.value = os.path.normpath(output_path.value)

		# Save config
		config['AUDIO'] = {
			'input': input_path.value,
			'output': output_path.value,
			'output_format': output_format.value,
			'normalize': normalize.value,
			'silent': silent.value,
		}
		config['PROCESS'] = {
			'music_1': music_1.value,
			'music_2': music_2.value,
			'vocal_1': vocal_1.value,
			'vocal_2': vocal_2.value,
			'bleed_1': bleed_1.value,
			'bleed_2': bleed_2.value,
			'bleed_3': bleed_3.value,
			'bleed_4': bleed_4.value,
			'bleed_5': bleed_5.value,
			'bleed_6': bleed_6.value,
		}
		config['OPTIONS'] = {
			'speed': speed.value,
			'chunk_size': chunk_size.value,
		}
		config['BONUS'] = {
			'KILL_on_END': KILL_on_END.value,
			'PREVIEWS': PREVIEWS.value,
			'DEBUG': DEBUG.value,
			'GOD_MODE': GOD_MODE.value,
			# TODO : Large GPU -> Do multiple Pass with steps with 3 models max for each Song
			# 'large_gpu': large_gpu.value,
			'large_gpu': False,
		}
		App.settings.Save(Gdrive, isColab, config)

		tab.selected_index = 1
		
		# Again the same bug for "tab titles" on Colab !!
		if isColab:
			display(HTML('<script type="application/javascript">show_titles();</script>'))

		params['input']		= []
		params['CONSOLE']	= CONSOLE
		params['Progress']	= Progress

		real_input  = os.path.join(Gdrive, input_path.value)

		if os.path.isfile(real_input):
			params['input'].append(real_input)
		else:
			# Get all audio files inside the folder (NOT recursive !)
			for file_path in sorted(glob.glob(os.path.join(real_input, "*.*")))[:]:
				if os.path.isfile(file_path):
					ext = os.path.splitext(file_path)[1].lower()
					if ext == ".mp3" or ext == ".wav" or ext == ".flac":
						params['input'].append(file_path)
		
		# Start processing
		if not Running:
			Running = True
			CONSOLE.clear_output();  App.inference.Process(params, config)
			Running = False


	def on_SysInfo_clicked(b):
		font_size = '13px' if isColab == True else '12px'
		sys_info.value = ""
		sys_info.value = App.sys_info.Get(font_size)

	# Delete all vocals files extracted from THIS song
	def on_Del_Vocals_clicked(b):

		# Get the folder based on input audio file's name
		name = os.path.splitext(os.path.basename(input_path.value))[0]

		deleted = ""		
		for file_path in sorted(glob.glob(os.path.join(Gdrive, output_path.value, name, "*.*")))[:]:
			filename = os.path.basename(file_path)
			if filename.startswith("4") or filename.startswith("5") or filename.startswith("6"):
				os.remove(file_path);  deleted += filename + ", "
		
		if deleted != "":
			deleted = deleted[:-2]  # Remove last ", "
			HELP.value = '<div id="HELP">Files deleted : '+ deleted +'</div>'
		else:
			HELP.value = '<div id="HELP"><div style="color: #f00">No files to delete !</div></div>'

	# Delete all music files extracted from THIS song
	def on_Del_Music_clicked(b):

		# Get the folder based on input audio file's name
		name = os.path.splitext(os.path.basename(input_path.value))[0]
		
		deleted = ""
		for file_path in sorted(glob.glob(os.path.join(Gdrive, output_path.value, name, "*.*")))[:]:
			filename = os.path.basename(file_path)
			if filename.startswith("2") or filename.startswith("3") or filename.startswith("4") or filename.startswith("5") or filename.startswith("6"):
				os.remove(file_path);  deleted += filename + ", "
		
		if deleted != "":
			deleted = deleted[:-2]  # Remove last ", "
			HELP.value = '<div id="HELP">Files deleted : '+ deleted +'</div>'
		else:
			HELP.value = '<div id="HELP"><div style="color: #f00">No files to delete !</div></div>'

	def on_Btn_Preset_1_clicked(b):
		music_1.value		= App.settings.Presets[0]['music_1']
		music_2.value		= App.settings.Presets[0]['music_2']
		vocal_1.value		= App.settings.Presets[0]['vocal_1']
		vocal_2.value		= App.settings.Presets[0]['vocal_2']
		bleed_1.value		= App.settings.Presets[0]['bleed_1']
		bleed_2.value		= App.settings.Presets[0]['bleed_2']
		bleed_3.value		= App.settings.Presets[0]['bleed_3']
		bleed_4.value		= App.settings.Presets[0]['bleed_4']
		bleed_5.value		= App.settings.Presets[0]['bleed_5']
		bleed_6.value		= App.settings.Presets[0]['bleed_6']
		
	def on_Btn_Preset_2_clicked(b):
		music_1.value		= App.settings.Presets[1]['music_1']
		music_2.value		= App.settings.Presets[1]['music_2']
		vocal_1.value		= App.settings.Presets[1]['vocal_1']
		vocal_2.value		= App.settings.Presets[1]['vocal_2']
		bleed_1.value		= App.settings.Presets[1]['bleed_1']
		bleed_2.value		= App.settings.Presets[1]['bleed_2']
		bleed_3.value		= App.settings.Presets[1]['bleed_3']
		bleed_4.value		= App.settings.Presets[1]['bleed_4']
		bleed_5.value		= App.settings.Presets[1]['bleed_5']
		bleed_6.value		= App.settings.Presets[1]['bleed_6']
		
	def on_Btn_Preset_3_clicked(b):
		music_1.value		= App.settings.Presets[2]['music_1']
		music_2.value		= App.settings.Presets[2]['music_2']
		vocal_1.value		= App.settings.Presets[2]['vocal_1']
		vocal_2.value		= App.settings.Presets[2]['vocal_2']
		bleed_1.value		= App.settings.Presets[2]['bleed_1']
		bleed_2.value		= App.settings.Presets[2]['bleed_2']
		bleed_3.value		= App.settings.Presets[2]['bleed_3']
		bleed_4.value		= App.settings.Presets[2]['bleed_4']
		bleed_5.value		= App.settings.Presets[2]['bleed_5']
		bleed_6.value		= App.settings.Presets[2]['bleed_6']
		
	def on_Btn_Preset_4_clicked(b):
		music_1.value		= App.settings.Presets[3]['music_1']
		music_2.value		= App.settings.Presets[3]['music_2']
		vocal_1.value		= App.settings.Presets[3]['vocal_1']
		vocal_2.value		= App.settings.Presets[3]['vocal_2']
		bleed_1.value		= App.settings.Presets[3]['bleed_1']
		bleed_2.value		= App.settings.Presets[3]['bleed_2']
		bleed_3.value		= App.settings.Presets[3]['bleed_3']
		bleed_4.value		= App.settings.Presets[3]['bleed_4']
		bleed_5.value		= App.settings.Presets[3]['bleed_5']
		bleed_6.value		= App.settings.Presets[3]['bleed_6']
		
	# Link Buttons to functions
	Btn_Del_Vocals.on_click(on_Del_Vocals_clicked)
	Btn_Del_Music.on_click(on_Del_Music_clicked)
	Btn_Preset_1.on_click(on_Btn_Preset_1_clicked)
	Btn_Preset_2.on_click(on_Btn_Preset_2_clicked)
	Btn_Preset_3.on_click(on_Btn_Preset_3_clicked)
	Btn_Preset_4.on_click(on_Btn_Preset_4_clicked)
	Btn_Start.on_click(on_Start_clicked)
	Btn_SysInfo.on_click(on_SysInfo_clicked)

	#**************
	#**  Events  **
	#**************
	
	def on_input_change(change):

		if HELP.value.find("ERROR") != -1:
			HELP.value = '<div id="HELP"></div>'  # Clear HELP

		# DO NOT USE os.path.normpath() :
		# it will remove the last separator in real-time --> impossible to type it in the input field !
		
		path = change['new']
		if path != "":
			path = path.replace('/', os.path.sep).replace('\\', os.path.sep)

			if path.find(Gdrive) != -1:
				path = path.replace(Gdrive, "")  # Remove Gdrive path from "input"
			
			# BUG signaled by Jarredou : remove the first separator
			if path[0] == os.path.sep:  path = path[1:]
			
			if path != input_path.value:  input_path.value = path
			
			if os.path.isdir(os.path.join(Gdrive, path)):
				HELP.value = '<div id="HELP"><span style="color: #c00000"><b>Your input is a folder path :</b><br>ALL audio files inside this folder will be separated by a Batch processing !</span></div>'
			else:
				HELP.value = '<div id="HELP"></div>'
		
		on_GOD_MODE_change(change)
		
	def on_output_change(change):
			
		if HELP.value.find("ERROR") != -1:
			HELP.value = '<div id="HELP"></div>'  # Clear HELP

		path = change['new']
		if path != "":
			path = path.replace('/', os.path.sep).replace('\\', os.path.sep)

			if path.find(Gdrive) != -1:
				path = path.replace(Gdrive, "")  # Remove Gdrive path from "output"

			# BUG signaled by Jarredou : remove the first separator
			if path[0] == os.path.sep:  path = path[1:]
			
			if path != output_path.value:  output_path.value = path
		
	def on_GOD_MODE_change(change):
		path = os.path.join(Gdrive, input_path.value)
		disable = not (os.path.isfile(path) and DEBUG.value and GOD_MODE.value)
		Btn_Del_Vocals.disabled = disable
		Btn_Del_Music.disabled  = disable

	def on_DEBUG_change(change):
		on_GOD_MODE_change(change)

	# Link Events to functions
	input_path.observe(on_input_change, names='value')
	output_path.observe(on_output_change, names='value')
	DEBUG.observe(on_DEBUG_change, names='value')
	GOD_MODE.observe(on_GOD_MODE_change, names='value')

	#*************
	#**  FINAL  **
	#*************

	javascript = '\
<script type="application/javascript">\
	var help_index = {};'
	
	for index in App.settings.Help_Dico.keys():
		javascript += f'\n	help_index["{index}"] = "{App.settings.Help_Dico[index]}";'

	# Correct the bug on Google Colab (no titles at all !!)
	if isColab:
		javascript += '\
function show_titles() {\
	document.getElementById("tab-key-0").getElementsByClassName("lm-TabBar-tabLabel")[0].innerHTML = "'+ titles[0] +'";\
	document.getElementById("tab-key-1").getElementsByClassName("lm-TabBar-tabLabel")[0].innerHTML = "'+ titles[1] +'";\
	document.getElementById("tab-key-2").getElementsByClassName("lm-TabBar-tabLabel")[0].innerHTML = "'+ titles[2] +'";\
}'

	# Add HELP
	javascript += '\
function show_help(index) {\
	document.getElementById("HELP").innerHTML = "<div>"+ help_index[index] +"</div>";\
}\
/* ... wait until the form is loaded */\
(function loop() {\
	setTimeout(() => {\
		if (document.getElementById("tab-key-0") == null || document.getElementById("HELP") == null) { loop(); return; }\
		document.getElementById("HELP").innerHTML = "<div style=\'color: #bbb\'>Hover your mouse over an option to get more informations.</div>";'
	
	if isColab:
		javascript += '\
		show_titles();\
		document.getElementById("tab-key-0").onclick = show_titles;\
		document.getElementById("tab-key-1").onclick = show_titles;\
		document.getElementById("tab-key-2").onclick = show_titles;\
		/* Auto-Scroll to KaraFan */\
		var lastCell = document.querySelector(".notebook-cell-list > div:last-child");\
		if (lastCell) window.location = document.URL.replace(/#.*/, "#scrollTo=" + lastCell.id.replace("cell-", ""));'
	
	javascript += '\
	}, 500);\
})();\
</script>'
	
	display(HTML(javascript))

	# Update controls after loading

	on_input_change({'new': input_path.value})
	on_output_change({'new': output_path.value})

	if config['PROCESS']['music_1'] in instru:		music_1.value = config['PROCESS']['music_1']
	if config['PROCESS']['music_2'] in instru:		music_2.value = config['PROCESS']['music_2']
	if config['PROCESS']['vocal_1'] in vocals:		vocal_1.value = config['PROCESS']['vocal_1']
	if config['PROCESS']['vocal_2'] in vocals:		vocal_2.value = config['PROCESS']['vocal_2']
	if config['PROCESS']['bleed_1'] in instru:		bleed_1.value = config['PROCESS']['bleed_1']
	if config['PROCESS']['bleed_2'] in instru:		bleed_2.value = config['PROCESS']['bleed_2']
	if config['PROCESS']['bleed_3'] in vocals:		bleed_3.value = config['PROCESS']['bleed_3']
	if config['PROCESS']['bleed_4'] in vocals:		bleed_4.value = config['PROCESS']['bleed_4']
	if config['PROCESS']['bleed_5'] in instru:		bleed_5.value = config['PROCESS']['bleed_5']
	if config['PROCESS']['bleed_6'] in instru:		bleed_6.value = config['PROCESS']['bleed_6']

	# DEBUG : Auto-start processing on execution
	if Auto_Start:  on_Start_clicked(None)
