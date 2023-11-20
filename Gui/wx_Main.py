
#   MIT License - Copyright (c) 2023 - Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan

import os, sys, csv, platform, threading, wx

import App.settings, Gui.wx_GPUtil, Gui.wx_Progress, Gui.wx_Window

# Change Font for ALL controls in a wxForm (Recursive)
def Set_Fonts(parent, font = None, color = None):

	for control in parent.GetChildren():

		if font is not None:
			if hasattr(control, "SetFont") and control.GetFont().GetFaceName() != "Tahoma" and not isinstance(control, wx.html.HtmlWindow):
				control.SetFont(font)

		if color is not None:
			if hasattr(control, "SetForegroundColour") and not isinstance(control, wx.html.HtmlWindow):
				control.SetForegroundColour(color)

		# Recursive
		if isinstance(control, wx.Notebook) or isinstance(control, wx.NotebookPage) \
		or isinstance(control, wx.BoxSizer) or isinstance(control, wx.StaticBox) \
		or isinstance(control, wx.FlexGridSizer) or isinstance(control, wx.WrapSizer):
			Set_Fonts(control, font)

class KaraFanForm(Gui.wx_Window.Form):

	def __init__(self, parent, params):
		Gui.wx_Window.Form.__init__(self, parent)

		# Set fonts for ALL controls
		if platform.system() == "Windows":
			Set_Fonts(self, font = wx.Font(11, wx.FONTFAMILY_DEFAULT, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False, "Segoe UI"))
		elif platform.system() == "Darwin":
			Set_Fonts(self, color = wx.Colour(0, 0, 0))

		self.params = params
		self.Gdrive = params['Gdrive']
		self.thread	= None
		self.timer	= None

		icon_path = params['Project'] + os.path.sep + "images" + os.path.sep

		# Reset defaults
		self.Tabs.SetSelection(0)
		self.Btn_Start.SetFocus()

		# Get local version
		with open(os.path.join(params['Project'], "App", "__init__.py"), "r") as version_file:
			Version = version_file.readline().replace("# Version", "").strip()

		self.SetTitle("KaraFan - " + Version)
		self.SetIcon(wx.Icon(icon_path + "KaraFan.ico", wx.BITMAP_TYPE_ICO))

		GPU = Gui.wx_GPUtil.getGPUs()
		if GPU:
			self.GPU.SetForegroundColour(wx.Colour(0, 179, 45))
			self.GPU.SetLabel(f"Using GPU ({(GPU[0].memoryTotal / 1024):.0f} GB) ►")  # Total amount of VRAM on the GPU (GB)

			self.timer = wx.Timer(self)  # Local Timer
			self.Bind(wx.EVT_TIMER, self.OnTimer, self.timer)  # Link timer to function
		else:
			if platform.system() == "Darwin":
				self.GPU.SetForegroundColour(wx.Colour(0, 179, 45))
				self.GPU.SetLabel(f"Using GPU ►")
				self.GPU_info.SetLabel("... if you have installed the correct drivers !")
			else:
				self.GPU.SetForegroundColour(wx.Colour(255, 0, 64))
				self.GPU.SetLabel("Using CPU ►►►")
				self.GPU_info.SetLabel("With CPU, processing will be very slow !!")

		# Set icons for each tab
		icon_music = wx.Image(icon_path + "icon-music.png", wx.BITMAP_TYPE_PNG).ConvertToBitmap()
		icon_vocal = wx.Image(icon_path + "icon-vocal.png", wx.BITMAP_TYPE_PNG).ConvertToBitmap()
		icon_list  = wx.ImageList(19, 19)
		icon_list.Add(wx.Image(icon_path + "icon-settings.png", wx.BITMAP_TYPE_PNG).ConvertToBitmap())
		icon_list.Add(wx.Image(icon_path + "icon-progress.png", wx.BITMAP_TYPE_PNG).ConvertToBitmap())
		icon_list.Add(wx.Image(icon_path + "icon-sys-info.png", wx.BITMAP_TYPE_PNG).ConvertToBitmap())
		self.Tabs.AssignImageList(icon_list)
		self.Tabs.SetPageImage(0, 0); self.Tabs.SetPageImage(1, 1); self.Tabs.SetPageImage(2, 2)

		self.icon1.SetBitmap(icon_music)
		self.icon2.SetBitmap(icon_vocal)
		self.icon3.SetBitmap(icon_music)
		self.icon4.SetBitmap(icon_vocal)
		self.icon5.SetBitmap(icon_music)

		# Set fonts for HTML controls
		self.CONSOLE.SetStandardFonts (12, "Courier New", "Courier New")
		self.sys_info.SetStandardFonts(10, "Courier New", "Courier New")

		self.html_start	= '<html><body text="#000000" bgcolor="#ffffd2"><font size="4">'
		self.html_end	= '</font></body></html>'
		
		# Get config values
		self.config = App.settings.Load(self.Gdrive, False)

		# Fill Models dropdowns
		vocals = ["----"]; instru = ["----"]
		with open(os.path.join(params['Project'], "Data", "Models.csv")) as csvfile:
			reader = csv.DictReader(csvfile)
			for row in reader:
				if row['Use'] == "x":
					# New fine-tuned MDX23C (less vocal bleedings in music ??)
					if row['Name'] == "MDX23C 8K FFT - v2" and not os.path.isfile(os.path.join(self.Gdrive, "KaraFan_user", "Models", "MDX23C-8KFFT-InstVoc_HQ_2.ckpt")):
						continue
					# ignore "Other" stems
					if row['Stem'] == "BOTH":			instru.append(row['Name']);  vocals.append(row['Name'])
					elif row['Stem'] == "Instrumental":	instru.append(row['Name'])
					elif row['Stem'] == "Vocals":		vocals.append(row['Name'])

		# Fill the Combo Boxes

		for item in App.settings.Options['Normalize']:	self.normalize.Append(item[0], item[1])
		for item in App.settings.Options['Format']:		self.output_format.Append(item[0], item[1])
		for item in App.settings.Options['Silent']:		self.silent.Append(item[0], item[1])

		for item in instru:
			self.music_1.Append(item); self.music_2.Append(item)
			self.bleed_1.Append(item); self.bleed_2.Append(item)
			self.bleed_5.Append(item); self.bleed_6.Append(item)
		for item in vocals:
			self.vocal_1.Append(item); self.vocal_2.Append(item)
			self.bleed_3.Append(item); self.bleed_4.Append(item)

		# TAB 1

		# AUDIO
		self.input_path.SetValue(self.config['AUDIO']['input'])   # Update HELP
		self.output_path.SetValue(self.config['AUDIO']['output']) # Update HELP
		self.infra_bass.Value = (self.config['AUDIO']['infra_bass'].lower() == "true")
		
		for index in range(len(App.settings.Options['Normalize'])):
			if self.config['AUDIO']['normalize'] == App.settings.Options['Normalize'][index][1]:  self.normalize.SetSelection(index); break
		for index in range(len(App.settings.Options['Format'])):
			if self.config['AUDIO']['output_format'] == App.settings.Options['Format'][index][1]:  self.output_format.SetSelection(index); break
		for index in range(len(App.settings.Options['Silent'])):
			if self.config['AUDIO']['silent'] == App.settings.Options['Silent'][index][1]:  self.silent.SetSelection(index); break
		
		# PROCESS
		if self.config['PROCESS']['music_1'] in instru:		self.music_1.SetStringSelection(self.config['PROCESS']['music_1'])
		else: self.music_1.SetSelection(0)
		if self.config['PROCESS']['music_2'] in instru:		self.music_2.SetStringSelection(self.config['PROCESS']['music_2'])
		else: self.music_2.SetSelection(0)
		if self.config['PROCESS']['vocal_1'] in vocals:		self.vocal_1.SetStringSelection(self.config['PROCESS']['vocal_1'])
		else: self.vocal_1.SetSelection(0)
		if self.config['PROCESS']['vocal_2'] in vocals:		self.vocal_2.SetStringSelection(self.config['PROCESS']['vocal_2'])
		else: self.vocal_2.SetSelection(0)
		if self.config['PROCESS']['bleed_1'] in instru:		self.bleed_1.SetStringSelection(self.config['PROCESS']['bleed_1'])
		else: self.bleed_1.SetSelection(0)
		if self.config['PROCESS']['bleed_2'] in instru:		self.bleed_2.SetStringSelection(self.config['PROCESS']['bleed_2'])
		else: self.bleed_2.SetSelection(0)
		if self.config['PROCESS']['bleed_3'] in vocals:		self.bleed_3.SetStringSelection(self.config['PROCESS']['bleed_3'])
		else: self.bleed_3.SetSelection(0)
		if self.config['PROCESS']['bleed_4'] in vocals:		self.bleed_4.SetStringSelection(self.config['PROCESS']['bleed_4'])
		else: self.bleed_4.SetSelection(0)
		if self.config['PROCESS']['bleed_5'] in instru:		self.bleed_5.SetStringSelection(self.config['PROCESS']['bleed_5'])
		else: self.bleed_5.SetSelection(0)
		if self.config['PROCESS']['bleed_6'] in instru:		self.bleed_6.SetStringSelection(self.config['PROCESS']['bleed_6'])
		else: self.bleed_6.SetSelection(0)

		self.high_pass.Value = int(self.config['PROCESS']['high_pass'])
		self.low_pass.Value  = int(self.config['PROCESS']['low_pass'])

		# OPTIONS
		for index in range(len(App.settings.Options['Speed'])):
			if self.config['OPTIONS']['speed'] == App.settings.Options['Speed'][index]:  self.speed.Value = index; break

		self.chunk_size.Value = int(self.config['OPTIONS']['chunk_size']) // 100000

		# BONUS
		self.DEBUG.Value		= (self.config['BONUS']['DEBUG'].lower() == "true")
		self.GOD_MODE.Value		= (self.config['BONUS']['GOD_MODE'].lower() == "true")
		self.KILL_on_END.Value	= (self.config['BONUS']['KILL_on_END'].lower() == "true")
		# TODO : Large GPU -> Do multiple Pass with steps with 3 models max for each Song
		# self.large_gpu.Value	= (self.config['BONUS']['large_gpu'].lower() == "true")
		
		self.HELP.SetPage(self.html_start +'<div style="color: #888">Hover your mouse over an option to get more informations.</div>'+ self.html_end)

		# TAB 2
		self.CONSOLE.SetPage("<b>Loading PyTorch, please wait ...</b><br>")
		self.Progress_Bar.Value = 0
		self.Progress_Text.SetLabel("")
		
		# TAB 3
		self.sys_info.SetPage("")

		# Update readouts text
		self.high_pass_OnSlider(None)
		self.low_pass_OnSlider(None)
		self.speed_OnSlider(None)
		self.chunk_size_OnSlider(None)

	#*********************
	#**  Buttons Click  **
	#*********************
	
	def Btn_Start_OnClick(self, event):

		self.HELP.SetPage(self.html_start + self.html_end)  # Clear HELP

		msg = ""
		if self.input_path.Value == "":
			msg += "Input is required !<br>"
		else:
			path = os.path.join(self.Gdrive, self.input_path.Value)
			if not os.path.isfile(path) and not os.path.isdir(path):
				msg += "Your Input is not a valid file or folder !<br>You MUST set it to an existing audio file or folder path.<br>"

		if self.output_path.Value == "":
			msg += "Output is required !<br>"
		else:
			path = os.path.join(self.Gdrive, self.output_path.Value)
			if not os.path.isdir(path):
				msg += "Your Output is not a valid folder !<br>You MUST set it to an existing folder path.<br>"
		
		if self.vocal_1.Value == "----" and self.vocal_2.Value == "----":
			msg += "You HAVE TO select at least one model for Vocals !<br>"
		
		if msg != "":
			msg = "ERROR !!<br>"+ msg
			self.HELP.SetPage(self.html_start +'<div style="color: #f00">'+ msg +'</div>'+ self.html_end)
			return
		
		# Normalize paths
		path = os.path.normpath(self.input_path.Value)
		if self.input_path.Value != path:	self.input_path.Value = path
		path = os.path.normpath(self.output_path.Value)
		if self.output_path.Value != path:	self.output_path.Value = path

		# Save config
		self.Form_OnClose(None)

		self.Tabs.ChangeSelection(1)
		
		self.params['CONSOLE']	= self.CONSOLE
		self.params['Progress']	= Gui.wx_Progress.Bar(self)  # Pass this wxForm to the Class for Progress Bar

		# Start processing
		if self.timer is not None:  self.timer.Start(2000)  # Interval : 2 sec.
		
		if self.thread is None or not self.thread.is_alive():

			self.thread = threading.Thread(target=self.Process)
			self.thread.start()

	def Process(self):
		import App.inference;  self.CONSOLE.SetPage("")
		
		try:
			App.inference.Process(self.params, self.config, wxWindow = self)  # Pass to "inference" this wxForm object

		except Exception as e:
			if not self.timer is None:  self.timer.Stop()
			
			Gui.wx_Error.Report(f"{e.__class__.__name__} : {e}", sys.exc_info()[2], self)


	def Btn_SysInfo_OnClick(self, event):
		import App.sys_info

		self.sys_info.SetPage("")
		self.sys_info.SetPage(App.sys_info.Get('14px'))

	def Btn_Preset_1_OnClick(self, event):
		self.music_1.Value		= App.settings.Presets[0]['music_1']
		self.music_2.Value		= App.settings.Presets[0]['music_2']
		self.vocal_1.Value		= App.settings.Presets[0]['vocal_1']
		self.vocal_2.Value		= App.settings.Presets[0]['vocal_2']
		self.bleed_1.Value		= App.settings.Presets[0]['bleed_1']
		self.bleed_2.Value		= App.settings.Presets[0]['bleed_2']
		self.bleed_3.Value		= App.settings.Presets[0]['bleed_3']
		self.bleed_4.Value		= App.settings.Presets[0]['bleed_4']
		self.bleed_5.Value		= App.settings.Presets[0]['bleed_5']
		self.bleed_6.Value		= App.settings.Presets[0]['bleed_6']

	def Btn_Preset_2_OnClick(self, event):
		self.music_1.Value		= App.settings.Presets[1]['music_1']
		self.music_2.Value		= App.settings.Presets[1]['music_2']
		self.vocal_1.Value		= App.settings.Presets[1]['vocal_1']
		self.vocal_2.Value		= App.settings.Presets[1]['vocal_2']
		self.bleed_1.Value		= App.settings.Presets[1]['bleed_1']
		self.bleed_2.Value		= App.settings.Presets[1]['bleed_2']
		self.bleed_3.Value		= App.settings.Presets[1]['bleed_3']
		self.bleed_4.Value		= App.settings.Presets[1]['bleed_4']
		self.bleed_5.Value		= App.settings.Presets[1]['bleed_5']
		self.bleed_6.Value		= App.settings.Presets[1]['bleed_6']

	def Btn_Preset_3_OnClick(self, event):
		self.music_1.Value		= App.settings.Presets[2]['music_1']
		self.music_2.Value		= App.settings.Presets[2]['music_2']
		self.vocal_1.Value		= App.settings.Presets[2]['vocal_1']
		self.vocal_2.Value		= App.settings.Presets[2]['vocal_2']
		self.bleed_1.Value		= App.settings.Presets[2]['bleed_1']
		self.bleed_2.Value		= App.settings.Presets[2]['bleed_2']
		self.bleed_3.Value		= App.settings.Presets[2]['bleed_3']
		self.bleed_4.Value		= App.settings.Presets[2]['bleed_4']
		self.bleed_5.Value		= App.settings.Presets[2]['bleed_5']
		self.bleed_6.Value		= App.settings.Presets[2]['bleed_6']

	def Btn_Preset_4_OnClick(self, event):
		self.music_1.Value		= App.settings.Presets[3]['music_1']
		self.music_2.Value		= App.settings.Presets[3]['music_2']
		self.vocal_1.Value		= App.settings.Presets[3]['vocal_1']
		self.vocal_2.Value		= App.settings.Presets[3]['vocal_2']
		self.bleed_1.Value		= App.settings.Presets[3]['bleed_1']
		self.bleed_2.Value		= App.settings.Presets[3]['bleed_2']
		self.bleed_3.Value		= App.settings.Presets[3]['bleed_3']
		self.bleed_4.Value		= App.settings.Presets[3]['bleed_4']
		self.bleed_5.Value		= App.settings.Presets[3]['bleed_5']
		self.bleed_6.Value		= App.settings.Presets[3]['bleed_6']

	def Btn_Preset_5_OnClick(self, event):
		self.music_1.Value		= App.settings.Presets[4]['music_1']
		self.music_2.Value		= App.settings.Presets[4]['music_2']
		self.vocal_1.Value		= App.settings.Presets[4]['vocal_1']
		self.vocal_2.Value		= App.settings.Presets[4]['vocal_2']
		self.bleed_1.Value		= App.settings.Presets[4]['bleed_1']
		self.bleed_2.Value		= App.settings.Presets[4]['bleed_2']
		self.bleed_3.Value		= App.settings.Presets[4]['bleed_3']
		self.bleed_4.Value		= App.settings.Presets[4]['bleed_4']
		self.bleed_5.Value		= App.settings.Presets[4]['bleed_5']
		self.bleed_6.Value		= App.settings.Presets[4]['bleed_6']

	def Btn_input_Path_OnClick( self, event ):
		dlg = wx.DirDialog(self, "Choose a folder :", "", wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
		if dlg.ShowModal() == wx.ID_OK:
			self.input_path.Value = dlg.GetPath().replace(self.Gdrive, "")

	def Btn_input_File_OnClick( self, event ):
		dlg = wx.FileDialog(self, "Choose a file :", "", "", "Audio files (*.flac;*.mp3;*.wav)|*.flac;*.mp3;*.wav", wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
		if dlg.ShowModal() == wx.ID_OK:
			self.input_path.Value = dlg.GetPath().replace(self.Gdrive, "")

	def Btn_output_Path_OnClick( self, event ):
		dlg = wx.DirDialog(self, "Choose a folder :", "", wx.DD_DEFAULT_STYLE | wx.DD_DIR_MUST_EXIST)
		if dlg.ShowModal() == wx.ID_OK:
			self.output_path.Value = dlg.GetPath().replace(self.Gdrive, "")

	#**************
	#**  Events  **
	#**************
	
	def Form_OnClose(self, event):

		self.config = {
			'AUDIO': {
				'input':		self.input_path.Value,
				'output':		self.output_path.Value,
				'output_format': self.output_format.GetClientData(self.output_format.GetSelection()),
				'normalize':	self.normalize.GetClientData(self.normalize.GetSelection()),
				'silent':		self.silent.GetClientData(self.silent.GetSelection()),
				'infra_bass':	self.infra_bass.Value,
			},
			'PROCESS': {
				'music_1': self.music_1.Value,
				'music_2': self.music_2.Value,
				'vocal_1': self.vocal_1.Value,
				'vocal_2': self.vocal_2.Value,
				'bleed_1': self.bleed_1.Value,
				'bleed_2': self.bleed_2.Value,
				'high_pass': self.high_pass.Value,
				'low_pass':  self.low_pass.Value,
				'bleed_3': self.bleed_3.Value,
				'bleed_4': self.bleed_4.Value,
				'bleed_5': self.bleed_5.Value,
				'bleed_6': self.bleed_6.Value,
			},
			'OPTIONS': {
				'speed': App.settings.Options['Speed'][self.speed.Value],
				'chunk_size': 	self.chunk_size.Value * 100000,
			},
			'BONUS': {
				'DEBUG':		self.DEBUG.Value,
				'GOD_MODE':		self.GOD_MODE.Value,
				'KILL_on_END': 	self.KILL_on_END.Value,
				# TODO : Large GPU -> Do multiple Pass with steps with 3 models max for each Song
				# 'large_gpu': large_gpu.Value,
				'large_gpu':	False,
			}
		}
		App.settings.Save(self.Gdrive, False, self.config)

		# If it's a REAL close event
		if not event is None:
			event.Skip()
			os._exit(0)  # Kill the process
	

	# Update GPU info
	def OnTimer(self, event):
		GPU = Gui.wx_GPUtil.getStatus()[0]
		if GPU:
			self.GPU_info.SetLabel(f"{GPU.memoryUsed / 1024:.2f} GB - ({GPU.memoryUtil * 100:.0f}%) - Load : {GPU.load * 100:.0f} % - Temp : {GPU.temperature:.0f} °C")

	def Show_Help(self, event):
		label = event.GetEventObject().GetName()
		if label in App.settings.Help_Dico:
			self.HELP.SetPage(self.html_start + App.settings.Help_Dico[label] + self.html_end)

	# Don't allow to type in the Combo Boxes (Better than "READONLY" that make the background color grey)
	def ComboBox_OnKeyDown(self, event):
		key = event.GetKeyCode()
		# but let UP / DOWN arrow & ALT-F4 & ENTER keys to work
		if key == 315 or key == 317 or key == 343 or key == 13:
			event.Skip()
		else:
			return
	
	def input_path_OnChange(self, event):

		if self.HELP.ToText().find("ERROR") != -1:
			self.HELP.SetPage(self.html_start + self.html_end)  # Clear HELP

		# DO NOT USE os.path.normpath() :
		# it will remove the last separator in real-time --> impossible to type it in the input field !
		
		path = self.input_path.Value
		if path != "":
			path = path.replace('/', os.path.sep).replace('\\', os.path.sep)

			if path != self.input_path.Value:  self.input_path.Value = path
			
			if os.path.isdir(os.path.join(self.Gdrive, path)):
				self.HELP.SetPage(self.html_start +'<span style="color: #c00000"><b>Your input is a folder path :</b><br>ALL audio files inside this folder will be separated by a Batch processing !</span>'+ self.html_end)
			else:
				self.HELP.SetPage(self.html_start + self.html_end)

	def output_path_OnChange(self, event):
			
		if self.HELP.ToText().find("ERROR") != -1:
			self.HELP.SetPage(self.html_start + self.html_end)  # Clear HELP

		path = self.output_path.Value
		if path != "":
			path = path.replace('/', os.path.sep).replace('\\', os.path.sep)

			if path != self.output_path.Value:  self.output_path.Value = path
		
	def high_pass_OnSlider(self, event):
		self.label_vocal_pass.SetLabel(f"Vocals Pass Band\n► {self.high_pass.Value * 5} Hz - {14 + (self.low_pass.Value * 0.5):.1f} KHz")

	def low_pass_OnSlider(self, event):
		self.high_pass_OnSlider(None)

	def speed_OnSlider(self, event):
		self.speed_readout.SetLabel(App.settings.Options['Speed'][self.speed.Value])

	def chunk_size_OnSlider(self, event):
		text = str(self.chunk_size.Value * 100) if self.chunk_size.Value < 10 else "1 000"
		self.chunk_size_readout.SetLabel(text + " 000")
