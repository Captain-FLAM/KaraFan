#!python3.10

#   MIT License - Copyright (c) 2023 - Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan

import os, csv, glob

import ipywidgets as widgets
from IPython.display import display, HTML

def Run(params, Auto_Start):

	import App.settings, App.inference, App.sys_info, App.progress

	Gdrive = params['Gdrive']
	Project = params['Project']
	isColab = params['isColab']
	DEV_MODE = params['I_AM_A_DEVELOPER']

	width  = '670px'
	height = '760px'

	# Set the font size when running on your PC
	font = '14px'
	font_small = '13px'

	# Set the font size when running on Colab
	if isColab:
		font = '15px';
		font_small = '13px'

	font_input = {'font_size': font}
	panel_layout = {'height': height, 'max_height': height, 'margin':'8px'}
	checkbox_layout = {'width': '50px', 'max_width': '50px' }
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
.path-info { font-size: '+ font +'; font-weight: bold }\
.path-warning { font-size: '+ font_small +'; font-style: italic; color: #c00000; margin: -3px 0 5px 0 }\
#HELP { font-size: '+ font +'; background-color: #ffffd2; border: solid 1px #333; width: 100%; height: 63px; line-height: 1.2 }\
#HELP > div { margin: 5px 10px }\
.console { font: normal '+ font +' monospace; line-height: 1.6 }\
.player { margin-bottom: 5px }\
.player > div { min-width: 200px; display: inline-block; font: normal '+ font +' monospace }\
.player > audio { vertical-align: middle }\
.SDR { display: inline-block; line-height: 1 }\
</style>'))
	
	def Label(text, index):
		return widgets.HTML(f'<div class="option-label" onmouseenter="show_help({index})">{text}</div>')
	
	# Get local version
	with open(os.path.join(Project, "App", "__init__.py"), "r") as version_file:
		Version = version_file.readline().replace("# Version", "").strip()

	# Get config values
	config = App.settings.Load(Gdrive, isColab)

	# Fill Models dropdowns
	vocals = ["----"]; instru = ["----"]; filters = ["----"]
	with open(os.path.join(Project, "App", "Models_DATA.csv")) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			if row['Use'] == "x":
				# ignore "Other" stems
				if row['Stem'] == "Instrumental":
					instru.append(row['Name'])
				elif row['Stem'] == "Vocals":
					vocals.append(row['Name']);  filters.append(row['Name'])  # Append ALSO "Vocals" to FILTERS !

	# KaraFan Title
	display(HTML('<div style="font-size: 24px; font-weight: bold; margin: 15px 0">KaraFan - version '+ Version +'</div>'))
	
	# TABS
	titles = ["☢️ Settings", "♾️ Progress", "❓ System Info"]

	# TAB 1
	separator		= widgets.HTML('<div style="border-bottom: dashed 1px #000; margin: 5px 0 5px 0; width: 100%">')
	# PATHS
	input_path		= widgets.Text(config['PATHS']['input'], continuous_update=True, style=font_input)
	output_path		= widgets.Text(config['PATHS']['output'], continuous_update=True, style=font_input)
	# PROCESS
	output_format	= widgets.Dropdown(value = config['PROCESS']['output_format'], options = App.settings.Options['Output_format'], layout = {'width':'150px'}, style=font_input)
	preset			= widgets.Dropdown(options = App.settings.Options['Presets'], layout = {'width':'150px'}, style=font_input)
	normalize		= widgets.Checkbox((config['PROCESS']['normalize'].lower() == "true"), indent=False, style=font_input, layout=checkbox_layout)
	vocals_1		= widgets.Dropdown(options = vocals, layout = {'width':'200px'}, style=font_input)
	vocals_2		= widgets.Dropdown(options = vocals, layout = {'width':'200px'}, style=font_input)
	vocals_3		= widgets.Dropdown(options = vocals, layout = {'width':'200px'}, style=font_input)
	vocals_4		= widgets.Dropdown(options = vocals, layout = {'width':'200px'}, style=font_input)
	REPAIR_MUSIC	= widgets.Dropdown(value = config['PROCESS']['REPAIR_MUSIC'], options = App.settings.Options['REPAIR_MUSIC'], layout = {'width':'200px'}, style=font_input)
	instru_1		= widgets.Dropdown(options = instru, layout = {'width':'200px'}, style=font_input)
	instru_2		= widgets.Dropdown(options = instru, layout = {'width':'200px'}, style=font_input)
	filter_1		= widgets.Dropdown(options = filters, layout = {'width':'200px'}, style=font_input)
	filter_2		= widgets.Dropdown(options = filters, layout = {'width':'200px'}, style=font_input)
	filter_3		= widgets.Dropdown(options = filters, layout = {'width':'200px'}, style=font_input)
	filter_4		= widgets.Dropdown(options = filters, layout = {'width':'200px'}, style=font_input)
	# OPTIONS
	speed			= widgets.SelectionSlider(value = config['OPTIONS']['speed'], options = App.settings.Options['Speed'], readout=True, style=font_input) 
	# overlap_MDXv3	= widgets.IntSlider(int(config['OPTIONS']['overlap_MDXv3']), min=2, max=40, step=2, layout={'margin':'0 0 0 10px'}, style=font_input)
	chunk_size		= widgets.IntSlider(int(config['OPTIONS']['chunk_size']), min=100000, max=1000000, step=100000, readout_format = ',d', style=font_input)
	# BONUS
	KILL_on_END		= widgets.Checkbox((config['BONUS']['KILL_on_END'].lower() == "true"), indent=False, style=font_input, layout=checkbox_layout)
	PREVIEWS		= widgets.Checkbox((config['BONUS']['PREVIEWS'].lower() == "true"), indent=False, style=font_input, layout=checkbox_layout)
	DEBUG			= widgets.Checkbox((config['BONUS']['DEBUG'].lower() == "true"), indent=False, continuous_update=True, style=font_input, layout=checkbox_layout)
	GOD_MODE		= widgets.Checkbox((config['BONUS']['GOD_MODE'].lower() == "true"), indent=False, continuous_update=True, style=font_input, layout=checkbox_layout)
	TEST_MODE		= widgets.Checkbox((config['BONUS']['TEST_MODE'].lower() == "true"), indent=False, style=font_input, layout=checkbox_layout)
	# TODO : Large GPU -> Do multiple Pass with steps with 3 models max for each Song
	# large_gpu		= widgets.Checkbox((config['BONUS']['large_gpu'].lower() == "true"), indent=False, style=font_input, layout=checkbox_layout)
	Btn_Del_Vocals	= widgets.Button(description='Vocals', button_style='danger', layout={'width':'80px', 'margin':'0 20px 0 0'})
	Btn_Del_Music	= widgets.Button(description='Music',  button_style='danger', layout={'width':'80px', 'margin':'0 20px 0 0'})
	# +
	HELP			= widgets.HTML('<div id="HELP"></div>')
	Btn_Start		= widgets.Button(description='Start', button_style='primary', layout={'width':'200px', 'margin':'12px 0 12px 0'})

	# TAB 2
	CONSOLE			= widgets.Output(layout = {'max_width': max_width, 'height': console_max_height, 'max_height': console_max_height, 'overflow':'scroll'})
	Progress_Bar	= widgets.IntProgress(value=0, min=0, max=10, orientation='horizontal', bar_style='info', layout={'width':'370px', 'height':'25px'})  # style={'bar_color': 'maroon'})
	Progress_Text	= widgets.HTML(layout={'width':'300px', 'margin':'0 0 0 15px'})
	# +
	Progress		= App.progress.Bar(Progress_Bar, Progress_Text)  # Class for Progress Bar

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
					widgets.HBox([ Label("Input X file or PATH", 101), input_path ]),
					widgets.HBox([ Label("Output PATH", 102), output_path ]),
				]),
				separator,
				widgets.VBox([
					widgets.HBox([ Label("Output Format", 201), output_format, Label("&nbsp; Normalize input", 202), normalize ]),
					widgets.HBox([ Label("Presets", 203), preset ]),
					widgets.HBox([ Label("MDX Vocals", 204),		vocals_1, vocals_3, widgets.HTML('<span style="font-size:18px">&nbsp; 💋</span>') ]),
					widgets.HBox([ Label("", 204),					vocals_2, vocals_4, widgets.HTML('<span style="font-size:18px">&nbsp; 💋</span>') ]),
				]),
				separator,
				widgets.VBox([
					widgets.HBox([ Label("Repair Music &nbsp;▶️", 205), REPAIR_MUSIC ]),
					widgets.HBox([ Label("MDX Music", 206),			instru_1, instru_2, widgets.HTML('<span style="font-size:18px">&nbsp; 🎵</span>') ]),
					widgets.HBox([ Label("MDX Music Clean", 207),	filter_1, filter_3, widgets.HTML('<span style="font-size:18px">&nbsp; ♒</span>') ]),
					widgets.HBox([ Label("", 207),					filter_2, filter_4, widgets.HTML('<span style="font-size:18px">&nbsp; ♒</span>') ]),
				]),
				separator,
				widgets.VBox([
					# TODO : Large GPU -> Do multiple Pass with steps with 2 models max for each Song
					widgets.HBox([ Label("Speed", 301), speed ]),
#					widgets.HBox([ Label("Overlap MDX v3", 302), overlap_MDXv3 ]),
					widgets.HBox([ Label("Chunk Size", 303), chunk_size ]),
				]),
				separator,
				widgets.VBox([
					widgets.HBox([ Label("This is the END ...", 401), KILL_on_END, Label("Show Previews", 402), PREVIEWS ]),
					widgets.HBox([ Label("DEBUG Mode", 403), DEBUG, Label("GOD Mode", 404), GOD_MODE, Label("TEST Mode", 405), TEST_MODE ]),
					# , Label('Large GPU', 406), large_gpu ]),
########					widgets.HBox([ Label("RE-Process ▶️▶️", 407), Btn_Del_Vocals, Btn_Del_Music ], layout={'margin':'15px 0 0 0'}),
				]),
				separator,
				widgets.HBox([Btn_Start], layout={'width':'100%', 'justify_content':'center'}),
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
	display(HTML('\
<script type="application/javascript">\
var help_index = []; help_index[1] = []; help_index[2] = []; help_index[3] = []; help_index[4] = [];\
help_index[1][1] = "- IF « Input » is a folder path, ALL audio files inside this folder will be separated by a Batch processing.<br>- Else, only the selected audio file will be processed.";\
help_index[1][2] = "« Output folder » will be created based on the file\'s name without extension.<br>For example : if your audio input is named : « 01 - Bohemian Rhapsody<b>.MP3</b> »,<br>then output folder will be named : « 01 - Bohemian Rhapsody »";\
help_index[2][1] = "Choose your prefered audio format to save audio files.";\
help_index[2][2] = "Normalize input audio files to avoid clipping and get better results.<br>Normally, <b>you do not have</b> to use this option !!<br>Only for weak or loud songs !";\
help_index[2][3] = "Genre of music to automatically select the best A.I models.";\
help_index[2][4] = "Make an Ensemble of extractions with Vocals selected models.<br><br>Best combination : « <b>Kim Vocal 2</b> » and « <b>Voc FT</b> »";\
help_index[2][5] = "Repair music with <b>A.I</b> models.<br>Use it if you hear missing instruments, but ... it will take <b>longer time</b> also !<br>If you hear too much <b>vocal bleedings in Music Final</b>, change Models or <b>DON\'T use it</b> !!";\
help_index[2][6] = "Make an Ensemble of instrumental extractions for repairing at the end of process.<br>Best combination : « <b>Inst HQ 3</b> » and ... test by yourself ! 😉<br>... You are warned : <b>ALL</b> instrumental models can carry <b>vocal bleedings</b> in final result !!";\
help_index[2][7] = "Pass Vocals trough different filters to remove <b>Vocals Bleedings</b> of instruments.<br>If you want to keep SFX in music, use only one model : « <b>Kim Vocal 1</b> » !<br>You have to test various models to find the best combination for your song ...";\
help_index[3][1] = "Set speed of extraction.<br>With fastest processing, don\'t complain about worst quality.<br><b>Slow</b> is the best quality, but it will take hours (days ? 😝) to process !!";\
help_index[3][2] = "MDX version 3 overlap. (default : 8)";\
help_index[3][3] = "Chunk size for ONNX models. (default : 500,000)<br><br>Set lower to reduce GPU memory consumption OR <b>if you have GPU memory errors</b> !";\
help_index[4][1] = "On <b>Colab</b> : KaraFan will KILL your session at end of « Processongs », to save your credits !!<br>On <b>your Laptop</b> : KaraFan will KILL your GPU, to save battery (and hot-less) !!<br>On <b>your PC</b> : KaraFan will KILL your GPU, anyway ... maybe it helps ? Try it !!";\
help_index[4][2] = "Shows an audio player for each saved file. For impatients people ! 😉<br><br>(Preview first 60 seconds with quality of MP3 - VBR 192 kbps)";\
help_index[4][3] = "IF checked, it will save all intermediate audio files to compare in your <b>Audacity</b>.";\
help_index[4][4] = "Give you the GOD\'s POWER : each audio file is reloaded IF it was created before,<br>NO NEED to process it again and again !!<br>You\'ll be warned : You have to <b>delete MANUALLY</b> each file that you want to re-process !";\
help_index[4][5] = "For <b>testing only</b> : Extract with A.I models with 1 pass instead of 2 passes.<br>The quality will be badder (due to weak noise added by MDX models) !<br>The normal <b>TWO PASSES</b> is the same as <b>DENOISE</b> option in <b>UVR 5</b> 😉";\
help_index[4][6] = "It will load ALL models in GPU memory for faster processing of MULTIPLE audio files.<br>Requires more GB of free GPU memory.<br>Uncheck it if you have memory troubles.";\
help_index[4][7] = "With <b>DEBUG</b> & <b>GOD MODE</b> activated : Available with <b>ONE file</b> at a time.<br>Automatic delete audio files of Stem that you want to re-process.<br>Vocals : <b>4_F</b> & <b>5_F</b> & <b>6</b>-Bleedings <b>/</b> Music : <b>same</b> + <b>2</b>-Music_extract & <b>3</b>-Audio_sub_Music";\
</script>'))

	# Bug in VS Code : titles NEEDS to be set AFTER children
	tab.titles = titles
	tab.selected_index = 0
	display(tab)

	#*********************
	#**  Buttons Click  **
	#*********************
	
	def on_Start_clicked(b):
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
		
		if vocals_1.value == "----" and vocals_2.value == "----" \
		and vocals_3.value == "----" and vocals_4.value == "----":
			msg += "You HAVE TO select at least one model for Vocals !<br>"
		
		if REPAIR_MUSIC.value and instru_1.value == "----" and instru_2.value == "----":
			msg += "You HAVE TO select at least one model for Instrumentals !<br>"

		if msg != "":
			msg = "ERROR !!<br>"+ msg
			HELP.value = '<div id="HELP"><div style="color: #f00">'+ msg +'</div></div>'
			return
		
		# Code from Deton24 : Keep session active !!
		if isColab:
			display(HTML('\
<script type="application/javascript">\
Keep_Running = setInterval(function() {\
	var selector = document.querySelector("#top-toolbar > colab-connect-button");\
	if (selector != null) {\
		selector.shadowRoot.querySelector("#connect").click();\
		setTimeout(function(selector) {\
			selector.shadowRoot.querySelector("#connect").click();\
		}, 2000 + Math.round(Math.random() * 6000));\
	}\
}, 120000);\
</script>'))

		input_path.value = os.path.normpath(input_path.value)
		output_path.value = os.path.normpath(output_path.value)

		# Save config
		config['PATHS'] = {
			'input': input_path.value,
			'output': output_path.value,
		}
		config['PROCESS'] = {
			'output_format': output_format.value,
			'preset': preset.value,
			'normalize': normalize.value,
			'vocals_1': vocals_1.value,
			'vocals_2': vocals_2.value,
			'vocals_3': vocals_3.value,
			'vocals_4': vocals_4.value,
			'REPAIR_MUSIC': REPAIR_MUSIC.value,
			'instru_1': instru_1.value,
			'instru_2': instru_2.value,
			'filter_1': filter_1.value,
			'filter_2': filter_2.value,
			'filter_3': filter_3.value,
			'filter_4': filter_4.value,
		}
		config['OPTIONS'] = {
			'speed': speed.value,
#			'overlap_MDXv3': overlap_MDXv3.value,
			'chunk_size': chunk_size.value,
		}
		config['BONUS'] = {
			'KILL_on_END': KILL_on_END.value,
			'PREVIEWS': PREVIEWS.value,
			'DEBUG': DEBUG.value,
			'GOD_MODE': GOD_MODE.value,
			'TEST_MODE': TEST_MODE.value,
			# TODO : Large GPU -> Do multiple Pass with steps with 3 models max for each Song
			# 'large_gpu': large_gpu.value,
			'large_gpu': False,
		}
		App.settings.Save(Gdrive, isColab, config)

		tab.selected_index = 1
		
		# Again the same bug for "tab titles" on Colab !!
		if isColab:
			display(HTML('<script type="application/javascript">show_titles();</script>'))

		params = {
			'input': [],
			'Gdrive': Gdrive,
			'Project': Project,
			'isColab': isColab,
			'CONSOLE': CONSOLE,
			'Progress': Progress,
			'DEV_MODE': DEV_MODE,
		}

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
		CONSOLE.clear_output();  App.inference.Process(params, config)

		# Refresh Google Drive files cache
#		if isColab:
#			from google.colab import drive
#			drive.flush_and_unmount(stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))
#			drive.mount("/content/Gdrive", stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))

		if isColab:
			# To stop automatic reactivation of Colab session
			display(HTML('<script type="application/javascript">clearInterval(Keep_Running);Keep_Running=null;</script>'))


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

	# Link Buttons to functions
	Btn_Del_Vocals.on_click(on_Del_Vocals_clicked)
	Btn_Del_Music.on_click(on_Del_Music_clicked)
	Btn_Start.on_click(on_Start_clicked)
	Btn_SysInfo.on_click(on_SysInfo_clicked)

	#**************
	#**  Events  **
	#**************
	
	def on_input_change(change):

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
			
		on_GOD_MODE_change(change);  update_paths_info()
		
	def on_output_change(change):
			
		path = change['new']
		if path != "":
			path = path.replace('/', os.path.sep).replace('\\', os.path.sep)

			if path.find(Gdrive) != -1:
				path = path.replace(Gdrive, "")  # Remove Gdrive path from "output"

			# BUG signaled by Jarredou : remove the first separator
			if path[0] == os.path.sep:  path = path[1:]
			
			if path != output_path.value:  output_path.value = path
		
		update_paths_info()
	
	def update_paths_info():

		if HELP.value.find("ERROR") != -1:
			HELP.value = '<div id="HELP"></div>'  # Clear HELP

		msg = ""
		path	= os.path.join(Gdrive, path)
		is_dir	= os.path.isdir(path)
		is_file	= os.path.isfile(path)
		
		if path != "":
			input_warning	= widgets.HTML('<div class="path-warning">Your input is a folder path :<br>ALL audio files inside this folder will be separated by a Batch processing !</div>')
			input_warning.layout.visibility = 'visible' if input_path.value != "" and is_dir else 'hidden'
		else:
			input_warning.layout.visibility = 'hidden'

			widgets.HBox([ widgets.HTML('<div class="option-label" style="color:#999">Your Final path</div>'), output_info ]),
		
		name = "[ NAME of FILES ]"
		if is_file:
			name = os.path.splitext(os.path.basename(path))[0]

		output_info.value = f'<div class="path-info">{Gdrive}{os.path.sep}{output_path.value} {os.path.sep} {name} {os.path.sep}</div>'

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
	var Keep_Running;'

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
	document.getElementById("HELP").innerHTML = "<div>"+ help_index[parseInt(index / 100)][index % 10] +"</div>";\
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

	# Example "Pop Rock" -> only one word, key = 'Rock'
	for words in App.settings.Options['Presets']:
		if config['PROCESS']['preset'] in words:  preset.value = words;  break

	if config['PROCESS']['vocals_1'] in vocals:		vocals_1.value = config['PROCESS']['vocals_1']
	if config['PROCESS']['vocals_2'] in vocals:		vocals_2.value = config['PROCESS']['vocals_2']
	if config['PROCESS']['vocals_3'] in vocals:		vocals_3.value = config['PROCESS']['vocals_3']
	if config['PROCESS']['vocals_4'] in vocals:		vocals_4.value = config['PROCESS']['vocals_4']
	if config['PROCESS']['instru_1'] in instru:		instru_1.value = config['PROCESS']['instru_1']
	if config['PROCESS']['instru_2'] in instru:		instru_2.value = config['PROCESS']['instru_2']
	if config['PROCESS']['filter_1'] in filters:	filter_1.value = config['PROCESS']['filter_1']
	if config['PROCESS']['filter_2'] in filters:	filter_2.value = config['PROCESS']['filter_2']
	if config['PROCESS']['filter_3'] in filters:	filter_3.value = config['PROCESS']['filter_3']
	if config['PROCESS']['filter_4'] in filters:	filter_4.value = config['PROCESS']['filter_4']


	# DEBUG : Auto-start processing on execution
	if Auto_Start:  on_Start_clicked(None)
