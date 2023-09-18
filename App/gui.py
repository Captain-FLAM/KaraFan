#!python3.10

#   MIT License - Copyright (c) 2023 Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan

import os, csv, glob

import ipywidgets as widgets
from IPython.display import display, HTML

def Run(Gdrive, Project, isColab):

	import App.settings, App.inference, App.sys_info

	width  = '670px'
	height = '720px'

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
	console_max_height = str(int(height.replace('px','')) - 18) + 'px'

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
</style>'))
	
	def Label(text, index):
		return widgets.HTML(f'<div class="option-label" onmouseenter="show_help({index})">{text}</div>')
	
	# Get local version
	with open(os.path.join(Project, "App", "__init__.py"), "r") as version_file:
		Version = version_file.readline().replace("# Version", "").strip()

	# Get config values
	config = App.settings.Load(Gdrive, isColab)

	# Fill Models dropdowns
	vocals = ["(None)"]; instru = ["(None)"]; filters = ["(None)"]
	with open(os.path.join(Project, "App", "Models_DATA.csv")) as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			# ignore "Other" stems
			# TODO : if row['Use'] != "":
			if row['Stem'] == "Instrumental":
				instru.append(row['Name']);  filters.append(row['Name'])  # Append "Instrumental" to FILTERS !
			elif row['Stem'] == "Vocals":
				vocals.append(row['Name']);  filters.append(row['Name'])  # Append ALSO "Vocals" to FILTERS !

	# KaraFan Title
	display(HTML('<div style="font-size: 24px; font-weight: bold; margin: 15px 0">KaraFan - version '+ Version +'</div>'))
	
	# TABS
	titles = ["☢️ Settings", "♾️ Progress", "❓ System Info"]

	# TAB 1
	separator = widgets.HTML('<div style="border-bottom: dashed 1px #000; margin: 5px 0 5px 0; width: 100%">')
	# PATHS
	input_path		= widgets.Text(config['PATHS']['input'], continuous_update=True, style=font_input)
	input_warning	= widgets.HTML('<div class="path-warning">Your input is a folder path : ALL audio files inside this folder will be separated by a Batch processing.</div>')
	output_path		= widgets.Text(config['PATHS']['output'], continuous_update=True, style=font_input)
	output_info		= widgets.HTML()
	Btn_Create_input  = widgets.Button(description='➕', tooltip="Create the input folder",  button_style='warning', layout={'display':'none', 'width':'25px', 'margin':'3px 0 0 15px', 'padding':'0'})
	Btn_Create_output = widgets.Button(description='➕', tooltip="Create the output folder", button_style='warning', layout={'display':'none', 'width':'25px', 'margin':'3px 0 0 15px', 'padding':'0'})
	# PROCESS
	output_format	= widgets.Dropdown(value = config['PROCESS']['output_format'], options=[("FLAC - 24 bits", "FLAC"), ("MP3 - CBR 320 kbps", "MP3"), ("WAV - PCM 16 bits","PCM_16"), ("WAV - FLOAT 32 bits","FLOAT")], layout = {'width':'200px'}, style=font_input)
	# preset_genre	= widgets.Dropdown(value = config['PROCESS']['preset_genre'], options=["Pop Rock"], disabled=True, layout = {'width':'150px'}, style=font_input)
	vocals_1		= widgets.Dropdown(options = vocals,  layout = {'width':'200px'}, style=font_input)
	vocals_2		= widgets.Dropdown(options = vocals,  layout = {'width':'200px'}, style=font_input)
	instru_1		= widgets.Dropdown(options = instru, layout = {'width':'200px'}, style=font_input)
	filter_1		= widgets.Dropdown(options = filters, layout = {'width':'200px'}, style=font_input)
	filter_2		= widgets.Dropdown(options = filters, layout = {'width':'200px'}, style=font_input)
	filter_3		= widgets.Dropdown(options = filters, layout = {'width':'200px'}, style=font_input)
	filter_4		= widgets.Dropdown(options = filters, layout = {'width':'200px'}, style=font_input)
	# OPTIONS
	normalize		= widgets.Checkbox((config['OPTIONS']['normalize'].lower() == "true"), indent=False, style=font_input, layout=checkbox_layout)
	large_gpu		= widgets.Checkbox((config['OPTIONS']['large_gpu'].lower() == "true"), indent=False, style=font_input, layout=checkbox_layout)
	shifts_vocals	= widgets.IntSlider(int(config['OPTIONS']['shifts_vocals']), min=1, max=24, step=1, style=font_input)
	shifts_instru	= widgets.IntSlider(int(config['OPTIONS']['shifts_instru']), min=1, max=24, step=1, style=font_input)
	shifts_filter	= widgets.IntSlider(int(config['OPTIONS']['shifts_filter']), min=1, max=12, step=1, style=font_input)
	# overlap_MDXv3	= widgets.IntSlider(int(config['OPTIONS']['overlap_MDXv3']), min=2, max=40, step=2, style=font_input)
	chunk_size		= widgets.IntSlider(int(config['OPTIONS']['chunk_size']), min=100000, max=1000000, step=100000, readout_format = ',d', style=font_input)
	# BONUS
	TEST_MODE		= widgets.Checkbox((config['BONUS']['TEST_MODE'].lower() == "true"), indent=False, style=font_input, layout=checkbox_layout)
	DEBUG			= widgets.Checkbox((config['BONUS']['DEBUG'].lower() == "true"), indent=False, continuous_update=True, style=font_input, layout=checkbox_layout)
	GOD_MODE		= widgets.Checkbox((config['BONUS']['GOD_MODE'].lower() == "true"), indent=False, continuous_update=True, style=font_input, layout=checkbox_layout)
	PREVIEWS		= widgets.Checkbox((config['BONUS']['PREVIEWS'].lower() == "true"), indent=False, style=font_input, layout=checkbox_layout)
	Btn_Del_Vocals	= widgets.Button(description='Vocals', button_style='danger', layout={'width':'80px', 'margin':'0 20px 0 0'})
	Btn_Del_Music	= widgets.Button(description='Music',  button_style='danger', layout={'width':'80px', 'margin':'0 20px 0 0'})
	#
	HELP			= widgets.HTML('<div id="HELP"></div>')
	Btn_Start		= widgets.Button(description='Start', button_style='primary', layout={'width':'200px', 'margin':'15px 0 15px 0'})

	# TAB 2
	Status			= widgets.Image(format='png', width=16, height=16, layout={'border': 'none', 'width':'16px', 'height':'16px', 'margin':'7px 0 0 0'})
	CONSOLE			= widgets.Output(layout = {'max_width': max_width, 'height': console_max_height, 'max_height': console_max_height, 'overflow':'scroll'})
	
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
					widgets.HBox([ Label("Input X file or PATH", 101), input_path, Btn_Create_input ]),
					input_warning,
					widgets.HBox([ Label("Output PATH", 102), output_path, Btn_Create_output ]),
					widgets.HBox([ widgets.HTML('<div class="option-label" style="color:#999">Your Final path</div>'), output_info ]),
				]),
				separator,
				widgets.VBox([
					widgets.HBox([ Label("Output Format", 201), output_format ]),
#					widgets.HBox([ Label("Preset Genre", 202), preset_genre, preset_models ]),
					widgets.HBox([ Label("MDX Vocals", 203), vocals_1, vocals_2, widgets.HTML('<span style="font-size:18px">&nbsp; 💋</span>') ]),
					widgets.HBox([ Label("MDX Instrumental", 204), instru_1, widgets.HTML('<span style="font-size:18px">&nbsp; 🎵</span>') ]),
					widgets.HBox([ Label("MDX Filters", 205), filter_1, filter_2, widgets.HTML('<span style="font-size:18px">&nbsp; ♒</span>') ]),
					widgets.HBox([ Label("", 205), filter_3, filter_4, widgets.HTML('<span style="font-size:18px">&nbsp; ♒</span>') ]),
				]),
				separator,
				widgets.VBox([
					widgets.HBox([ Label("Normalize input", 301), normalize, Label('Large GPU', 302), large_gpu ]),
					widgets.HBox([ Label("BigShifts Vocals", 303),  shifts_vocals ]),
					widgets.HBox([ Label("BigShifts Instrum", 303), shifts_instru ]),
					widgets.HBox([ Label("BigShifts Filters", 303), shifts_filter ]),
#					widgets.HBox([ Label("Overlap MDX v3", 304), overlap_MDXv3 ]),
					widgets.HBox([ Label("Chunk Size", 305), chunk_size ]),
				]),
				separator,
				widgets.VBox([
					widgets.HBox([ Label("DEBUG", 401), DEBUG, Label("TEST Mode", 402), TEST_MODE ]),
					widgets.HBox([ Label("GOD Mode", 403), GOD_MODE, Label("Show Previews", 404), PREVIEWS ]),
########					widgets.HBox([ Label("RE-Process ▶️▶️", 405), Btn_Del_Vocals, Btn_Del_Music ], layout={'margin':'15px 0 0 0'}),
				]),
				separator,
				widgets.HBox([Btn_Start], layout={'width':'100%', 'justify_content':'center'}),
				HELP
			]),
		widgets.VBox(
			layout = panel_layout,
			children = [
				widgets.HBox([ widgets.HTML('<span>Status : &nbsp;</span>'), Status ]),
				CONSOLE
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
help_index[2][2] = "Genre of music to automatically select the best A.I models.";\
help_index[2][3] = "<b>A.I</b> models : Make an Ensemble of extraction with selected models.<br><br>Best combination : « <b>Kim Vocal 2</b> » and « <b>Voc FT</b> »";\
help_index[2][4] = "<b>A.I</b> models : Extract the instrumental part for repairing at the end of process.<br><br>Best model : « <b>Inst HQ 3</b> »";\
help_index[2][5] = "<b>A.I</b> models : Pass Vocals trough different filters to remove <b>Bleedings</b> of instruments.<br><br>You have to test various models to find the best combination for your song !";\
help_index[3][1] = "Normalize input audio files to avoid clipping and get better results.<br><br>Uncheck it for <b>SDR</b> testings !!";\
help_index[3][2] = "It will load ALL models in GPU memory for faster processing of MULTIPLE audio files.<br>Requires more GB of free GPU memory.<br>Uncheck it if you have memory troubles.";\
help_index[3][3] = "Set MDX « BigShifts » trick value. (default : 12 , filters : 2)<br><br>Set it to = 1 to disable that feature.";\
help_index[3][4] = "MDX version 3 overlap. (default : 8)";\
help_index[3][5] = "Chunk size for ONNX models. (default : 500,000)<br><br>Set lower to reduce GPU memory consumption OR <b>if you have GPU memory errors</b> !";\
help_index[4][1] = "IF checked, it will save all intermediate audio files to compare in your <b>Audacity</b>.";\
help_index[4][2] = "For <b>testing only</b> : Extract with A.I models with 1 pass instead of 2 passes.<br>The quality will be badder (due to weak noise added by MDX models) !<br>The normal <b>TWO PASSES</b> is the same as <b>DENOISE</b> option in <b>UVR 5</b> 😉";\
help_index[4][3] = "Give you the GOD\'s POWER : each audio file is reloaded IF it was created before,<br>NO NEED to process it again and again !!<br>You\'ll be warned : You have to <b>delete MANUALLY</b> each file that you want to re-process !";\
help_index[4][4] = "Shows an audio player for each saved file. For impatients people ! 😉<br><br>(Preview first 60 seconds with quality of MP3 - VBR 192 kbps)";\
help_index[4][5] = "With <b>DEBUG</b> & <b>GOD MODE</b> activated : Available with <b>ONE file</b> at a time.<br>Automatic delete audio files of Stem that you want to re-process.<br>Vocals : <b>4_F</b> & <b>5_F</b> & <b>6</b>-Bleedings <b>/</b> Music : <b>same</b> + <b>2</b>-Music_extract & <b>3</b>-Audio_sub_Music";\
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
		
		if msg != "":
			msg = "ERROR !!<br>"+ msg
			HELP.value = '<div id="HELP"><div style="color: #f00">'+ msg +'</div></div>'
			return
		
		input_path.value = os.path.normpath(input_path.value)
		output_path.value = os.path.normpath(output_path.value)

		# Save config
		config['PATHS'] = {
			'input': input_path.value,
			'output': output_path.value,
		}
		config['PROCESS'] = {
			'output_format': output_format.value,
#			'preset_genre': preset_genre.value,
			'vocals_1': vocals_1.value,
			'vocals_2': vocals_2.value,
			'instru_1': instru_1.value,
			'filter_1': filter_1.value,
			'filter_2': filter_2.value,
			'filter_3': filter_3.value,
			'filter_4': filter_4.value,
		}
		config['OPTIONS'] = {
			'shifts_vocals': shifts_vocals.value,
			'shifts_instru': shifts_instru.value,
			'shifts_filter': shifts_filter.value,
#			'overlap_MDXv3': overlap_MDXv3.value,
			'chunk_size': chunk_size.value,
			'normalize': normalize.value,
			'large_gpu': large_gpu.value,
		}
		config['BONUS'] = {
			'TEST_MODE': TEST_MODE.value,
			'DEBUG': DEBUG.value,
			'GOD_MODE': GOD_MODE.value,
			'PREVIEWS': PREVIEWS.value,
		}
		App.settings.Save(Gdrive, isColab, config)

		tab.selected_index = 1
		
		# Again the same bug for "tab titles" on Colab !!
		if isColab:
			display(HTML('<script type="application/javascript">show_titles();</script>'))

		options = App.settings.Convert_to_Options(config)

		options['Gdrive'] = Gdrive
		options['Project'] = Project
		options['CONSOLE'] = CONSOLE
		options['Status']  = Status
		
		options['input'] = []

		real_input  = os.path.join(Gdrive, input_path.value)

		if os.path.isfile(real_input):
			options['input'].append(real_input)
		else:
			# Get all audio files inside the folder (NOT recursive !)
			for file_path in sorted(glob.glob(os.path.join(real_input, "*.*")))[:]:
				if os.path.isfile(file_path):
					ext = os.path.splitext(file_path)[1].lower()
					if ext == ".mp3" or ext == ".wav" or ext == ".flac":
						options['input'].append(file_path)
		
		# Start processing
		CONSOLE.clear_output();  App.inference.Process(options)

		# Refresh Google Drive files cache
#		if isColab:
#			from google.colab import drive
#			drive.flush_and_unmount(stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))
#			drive.mount("/content/Gdrive", stdout=open('/dev/null', 'w'), stderr=open('/dev/null', 'w'))


	def on_SysInfo_clicked(b):
		font_size = '13px' if isColab == True else '12px'
		Btn_SysInfo.layout = {'display':'none'}
		sys_info.value = App.sys_info.Get(font_size)

	def on_Create_input_clicked(b):
		os.makedirs(os.path.join(Gdrive, input_path.value), exist_ok=True)
		Btn_Create_input.layout.display = 'none'
		on_input_change({'new': input_path.value})

	def on_Create_output_clicked(b):
		os.makedirs(os.path.join(Gdrive, output_path.value), exist_ok=True)
		Btn_Create_output.layout.display = 'none'
		on_output_change({'new': output_path.value})
	
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
	Btn_Create_input.on_click(on_Create_input_clicked)
	Btn_Create_output.on_click(on_Create_output_clicked)
	Btn_Del_Vocals.on_click(on_Del_Vocals_clicked)
	Btn_Del_Music.on_click(on_Del_Music_clicked)
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
			if path[0] == os.path.sep:
				path = path[1:]
			
			if path != input_path.value:  input_path.value = path
			
		path	= os.path.join(Gdrive, path)
		is_dir	= os.path.isdir(path)
		is_file	= os.path.isfile(path)
		
		if path != "":
			input_warning.layout.display = 'block' if input_path.value != "" and is_dir else 'none'
			Btn_Create_input.layout.display = 'inline' if not is_file and not is_dir else 'none'
		else:
			Btn_Create_input.layout.display = 'none'
			input_warning.layout.display = 'none'

		on_GOD_MODE_change(change)

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

	def on_output_change(change):
		if HELP.value.find("ERROR") != -1:
			HELP.value = '<div id="HELP"></div>'  # Clear HELP
			
		path = change['new']

		if path != "":
			path = path.replace('/', os.path.sep).replace('\\', os.path.sep)

			if path.find(Gdrive) != -1:
				path = path.replace(Gdrive, "")  # Remove Gdrive path from "output"

			# BUG signaled by Jarredou : remove the first separator
			if path[0] == os.path.sep:
				path = path[1:]
			
			if path != output_path.value:  output_path.value = path

			Btn_Create_output.layout.display = 'inline' if not os.path.isdir(os.path.join(Gdrive, path)) else 'none'
		else:
			Btn_Create_output.layout.display = 'none'

	# Link Events to functions
	input_path.observe(on_input_change, names='value')
	output_path.observe(on_output_change, names='value')
	DEBUG.observe(on_DEBUG_change, names='value')
	GOD_MODE.observe(on_GOD_MODE_change, names='value')

	#*************
	#**  FINAL  **
	#*************

	javascript = '\
<script type="application/javascript">'
	
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
		\
		document.getElementById("HELP").innerHTML = "<div style=\'color: #bbb\'>Hover your mouse over an option to get more informations.</div>";'
	
	if isColab:
		javascript += '\
		show_titles();\
		document.getElementById("tab-key-0").onclick = show_titles;\
		document.getElementById("tab-key-1").onclick = show_titles;\
		document.getElementById("tab-key-2").onclick = show_titles;'
	
	javascript += '\
	}, 500);\
})();\
</script>'
	
	display(HTML(javascript))

	# Update controls after loading

	on_input_change({'new': input_path.value})
	on_output_change({'new': output_path.value})

	# Load Status Led icons
	with open(os.path.join(Project, "images", "Led_Grey.png"), 'rb') as file:
		Status.value = file.read()

	if config['PROCESS']['vocals_1'] in vocals:		vocals_1.value = config['PROCESS']['vocals_1']
	if config['PROCESS']['vocals_2'] in vocals:		vocals_2.value = config['PROCESS']['vocals_2']
	if config['PROCESS']['instru_1'] in instru:		instru_1.value = config['PROCESS']['instru_1']
	if config['PROCESS']['filter_1'] in filters:	filter_1.value = config['PROCESS']['filter_1']
	if config['PROCESS']['filter_2'] in filters:	filter_2.value = config['PROCESS']['filter_2']
	if config['PROCESS']['filter_3'] in filters:	filter_3.value = config['PROCESS']['filter_3']
	if config['PROCESS']['filter_4'] in filters:	filter_4.value = config['PROCESS']['filter_4']
