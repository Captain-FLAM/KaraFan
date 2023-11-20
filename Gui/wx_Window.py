# -*- coding: utf-8 -*-

###########################################################################
## Python code generated with wxFormBuilder (version 3.10.1-0-g8feb16b3)
## http://www.wxformbuilder.org/
##
## PLEASE DO *NOT* EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc
import wx.html

import gettext
_ = gettext.gettext

###########################################################################
## Class Form
###########################################################################

class Form ( wx.Frame ):

	def __init__( self, parent ):
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = _(u"KaraFan"), pos = wx.DefaultPosition, size = wx.Size( 697,806 ), style = wx.DEFAULT_FRAME_STYLE|wx.TAB_TRAVERSAL )

		self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )
		self.SetBackgroundColour( wx.Colour( 255, 255, 255 ) )

		MainSizer = wx.BoxSizer( wx.VERTICAL )

		self.Tabs = wx.Notebook( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.Tabs.SetFont( wx.Font( 12, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False, "Arial" ) )

		self.Tab_Settings = wx.Panel( self.Tabs, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		box1 = wx.BoxSizer( wx.VERTICAL )

		sbSizer1 = wx.StaticBoxSizer( wx.StaticBox( self.Tab_Settings, wx.ID_ANY, _(u"[ Songs ]") ), wx.VERTICAL )

		bSizer7 = wx.BoxSizer( wx.HORIZONTAL )

		bSizer8 = wx.BoxSizer( wx.HORIZONTAL )

		bSizer8.SetMinSize( wx.Size( 150,-1 ) )
		self.label_input = wx.StaticText( sbSizer1.GetStaticBox(), wx.ID_ANY, _(u"Input"), wx.DefaultPosition, wx.Size( 75,-1 ), 0, u"input" )
		self.label_input.Wrap( -1 )

		bSizer8.Add( self.label_input, 0, wx.TOP, 5 )

		self.Btn_input_Path = wx.Button( sbSizer1.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.BORDER_NONE|wx.BU_EXACTFIT )

		self.Btn_input_Path.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_FOLDER_OPEN, wx.ART_HELP_BROWSER ) )
		self.Btn_input_Path.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOW ) )
		self.Btn_input_Path.SetToolTip( _(u"PATH") )

		bSizer8.Add( self.Btn_input_Path, 0, wx.LEFT|wx.RIGHT, 5 )

		self.Btn_input_File = wx.Button( sbSizer1.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.BORDER_NONE|wx.BU_EXACTFIT )

		self.Btn_input_File.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_NORMAL_FILE, wx.ART_HELP_BROWSER ) )
		self.Btn_input_File.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOW ) )
		self.Btn_input_File.SetToolTip( _(u"X File") )

		bSizer8.Add( self.Btn_input_File, 0, wx.LEFT|wx.RIGHT, 5 )


		bSizer7.Add( bSizer8, 0, wx.EXPAND, 5 )

		self.input_path = wx.TextCtrl( sbSizer1.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 320,-1 ), 0 )
		bSizer7.Add( self.input_path, 0, wx.BOTTOM|wx.RIGHT, 5 )

		self.label_normalize = wx.StaticText( sbSizer1.GetStaticBox(), wx.ID_ANY, _(u"Normalize "), wx.DefaultPosition, wx.DefaultSize, 0, u"normalize" )
		self.label_normalize.Wrap( -1 )

		bSizer7.Add( self.label_normalize, 0, wx.LEFT|wx.TOP, 5 )

		normalizeChoices = []
		self.normalize = wx.ComboBox( sbSizer1.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 73,-1 ), normalizeChoices, 0 )
		bSizer7.Add( self.normalize, 0, 0, 5 )


		sbSizer1.Add( bSizer7, 1, wx.EXPAND|wx.LEFT, 5 )

		fgSizer6 = wx.FlexGridSizer( 0, 6, 0, 0 )
		fgSizer6.SetFlexibleDirection( wx.BOTH )
		fgSizer6.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )

		bSizer9 = wx.BoxSizer( wx.HORIZONTAL )

		bSizer9.SetMinSize( wx.Size( 150,-1 ) )
		self.label_output = wx.StaticText( sbSizer1.GetStaticBox(), wx.ID_ANY, _(u"Output "), wx.DefaultPosition, wx.Size( 75,-1 ), 0, u"output" )
		self.label_output.Wrap( -1 )

		bSizer9.Add( self.label_output, 0, wx.TOP, 5 )

		self.Btn_output_Path = wx.Button( sbSizer1.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.BORDER_NONE|wx.BU_EXACTFIT )

		self.Btn_output_Path.SetBitmap( wx.ArtProvider.GetBitmap( wx.ART_FOLDER_OPEN, wx.ART_HELP_BROWSER ) )
		self.Btn_output_Path.SetBackgroundColour( wx.SystemSettings.GetColour( wx.SYS_COLOUR_WINDOW ) )
		self.Btn_output_Path.SetToolTip( _(u"PATH") )

		bSizer9.Add( self.Btn_output_Path, 0, wx.LEFT|wx.RIGHT, 5 )


		fgSizer6.Add( bSizer9, 0, wx.EXPAND, 5 )

		self.output_path = wx.TextCtrl( sbSizer1.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 320,-1 ), 0 )
		fgSizer6.Add( self.output_path, 0, 0, 5 )


		sbSizer1.Add( fgSizer6, 0, wx.BOTTOM|wx.EXPAND|wx.LEFT, 5 )

		fgSizer71 = wx.FlexGridSizer( 0, 6, 0, 0 )
		fgSizer71.SetFlexibleDirection( wx.BOTH )
		fgSizer71.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )

		self.label_format = wx.StaticText( sbSizer1.GetStaticBox(), wx.ID_ANY, _(u"Output Format"), wx.DefaultPosition, wx.Size( 150,-1 ), 0, u"format" )
		self.label_format.Wrap( -1 )

		fgSizer71.Add( self.label_format, 0, wx.TOP, 5 )

		output_formatChoices = []
		self.output_format = wx.ComboBox( sbSizer1.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 160,-1 ), output_formatChoices, 0 )
		fgSizer71.Add( self.output_format, 0, wx.RIGHT, 20 )

		self.label_silent = wx.StaticText( sbSizer1.GetStaticBox(), wx.ID_ANY, _(u"Silent"), wx.DefaultPosition, wx.DefaultSize, 0, u"silent" )
		self.label_silent.Wrap( -1 )

		fgSizer71.Add( self.label_silent, 0, wx.TOP, 5 )

		silentChoices = []
		self.silent = wx.ComboBox( sbSizer1.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 96,-1 ), silentChoices, 0 )
		self.silent.SetBackgroundColour( wx.Colour( 255, 255, 255 ) )

		fgSizer71.Add( self.silent, 0, wx.LEFT|wx.RIGHT, 5 )

		self.label_infra_bass = wx.StaticText( sbSizer1.GetStaticBox(), wx.ID_ANY, _(u"KILL Infra-Bass"), wx.DefaultPosition, wx.DefaultSize, 0, u"infra_bass" )
		self.label_infra_bass.Wrap( -1 )

		fgSizer71.Add( self.label_infra_bass, 0, wx.LEFT|wx.RIGHT|wx.TOP, 5 )

		self.infra_bass = wx.CheckBox( sbSizer1.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		fgSizer71.Add( self.infra_bass, 0, wx.LEFT|wx.RIGHT|wx.TOP, 8 )


		sbSizer1.Add( fgSizer71, 0, wx.BOTTOM|wx.EXPAND|wx.LEFT, 5 )


		box1.Add( sbSizer1, 0, wx.ALL|wx.EXPAND, 5 )

		sbSizer2 = wx.StaticBoxSizer( wx.StaticBox( self.Tab_Settings, wx.ID_ANY, _(u"[ Vocals ]") ), wx.VERTICAL )

		fgSizer7 = wx.FlexGridSizer( 0, 4, 0, 0 )
		fgSizer7.SetFlexibleDirection( wx.BOTH )
		fgSizer7.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )

		self.label_MDX_music = wx.StaticText( sbSizer2.GetStaticBox(), wx.ID_ANY, _(u"Filter Music"), wx.DefaultPosition, wx.Size( 150,-1 ), 0, u"MDX_music" )
		self.label_MDX_music.Wrap( -1 )

		fgSizer7.Add( self.label_MDX_music, 0, wx.TOP, 5 )

		music_1Choices = []
		self.music_1 = wx.ComboBox( sbSizer2.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 220,-1 ), music_1Choices, 0 )
		fgSizer7.Add( self.music_1, 0, wx.BOTTOM, 5 )

		music_2Choices = []
		self.music_2 = wx.ComboBox( sbSizer2.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 220,-1 ), music_2Choices, 0 )
		fgSizer7.Add( self.music_2, 0, wx.LEFT|wx.RIGHT, 10 )

		self.icon1 = wx.StaticBitmap( sbSizer2.GetStaticBox(), wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, 0 )
		fgSizer7.Add( self.icon1, 0, wx.TOP, 3 )

		self.label_MDX_vocal = wx.StaticText( sbSizer2.GetStaticBox(), wx.ID_ANY, _(u"Extract Vocals"), wx.DefaultPosition, wx.Size( 150,-1 ), 0, u"MDX_vocal" )
		self.label_MDX_vocal.Wrap( -1 )

		fgSizer7.Add( self.label_MDX_vocal, 0, wx.TOP, 5 )

		vocal_1Choices = []
		self.vocal_1 = wx.ComboBox( sbSizer2.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 220,-1 ), vocal_1Choices, 0 )
		fgSizer7.Add( self.vocal_1, 0, wx.BOTTOM, 5 )

		vocal_2Choices = []
		self.vocal_2 = wx.ComboBox( sbSizer2.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 220,-1 ), vocal_2Choices, 0 )
		fgSizer7.Add( self.vocal_2, 0, wx.LEFT|wx.RIGHT, 10 )

		self.icon2 = wx.StaticBitmap( sbSizer2.GetStaticBox(), wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, 0 )
		fgSizer7.Add( self.icon2, 0, wx.TOP, 3 )

		self.label_MDX_bleed_1 = wx.StaticText( sbSizer2.GetStaticBox(), wx.ID_ANY, _(u"Music Bleedings"), wx.DefaultPosition, wx.Size( 150,-1 ), 0, u"MDX_bleed_1" )
		self.label_MDX_bleed_1.Wrap( -1 )

		fgSizer7.Add( self.label_MDX_bleed_1, 0, wx.TOP, 5 )

		bleed_1Choices = []
		self.bleed_1 = wx.ComboBox( sbSizer2.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 220,-1 ), bleed_1Choices, 0 )
		fgSizer7.Add( self.bleed_1, 0, wx.BOTTOM, 5 )

		bleed_2Choices = []
		self.bleed_2 = wx.ComboBox( sbSizer2.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 220,-1 ), bleed_2Choices, 0 )
		fgSizer7.Add( self.bleed_2, 0, wx.LEFT|wx.RIGHT, 10 )

		self.icon3 = wx.StaticBitmap( sbSizer2.GetStaticBox(), wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, 0 )
		fgSizer7.Add( self.icon3, 0, wx.TOP, 3 )

		self.label_vocal_pass = wx.StaticText( sbSizer2.GetStaticBox(), wx.ID_ANY, _(u"Vocals Pass Band\n► 0 Hz - 22.0 KHz"), wx.DefaultPosition, wx.Size( 150,-1 ), 0, u"vocal_pass" )
		self.label_vocal_pass.Wrap( -1 )

		fgSizer7.Add( self.label_vocal_pass, 0, wx.TOP, 5 )

		self.high_pass = wx.Slider( sbSizer2.GetStaticBox(), wx.ID_ANY, 14, 0, 20, wx.DefaultPosition, wx.Size( 216,-1 ), wx.SL_AUTOTICKS|wx.SL_HORIZONTAL )
		fgSizer7.Add( self.high_pass, 0, wx.ALL|wx.EXPAND, 2 )

		self.low_pass = wx.Slider( sbSizer2.GetStaticBox(), wx.ID_ANY, 5, 0, 16, wx.DefaultPosition, wx.Size( 216,-1 ), wx.SL_AUTOTICKS|wx.SL_HORIZONTAL )
		fgSizer7.Add( self.low_pass, 0, wx.ALL|wx.EXPAND, 2 )


		sbSizer2.Add( fgSizer7, 0, wx.BOTTOM|wx.EXPAND|wx.LEFT, 5 )


		box1.Add( sbSizer2, 0, wx.BOTTOM|wx.EXPAND|wx.LEFT|wx.RIGHT, 5 )

		sbSizer21 = wx.StaticBoxSizer( wx.StaticBox( self.Tab_Settings, wx.ID_ANY, _(u"[  Music  ]") ), wx.VERTICAL )

		fgSizer911 = wx.FlexGridSizer( 0, 4, 0, 0 )
		fgSizer911.SetFlexibleDirection( wx.BOTH )
		fgSizer911.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )

		self.label_MDX_bleed_2 = wx.StaticText( sbSizer21.GetStaticBox(), wx.ID_ANY, _(u"Vocal Bleedings"), wx.DefaultPosition, wx.Size( 150,-1 ), 0, u"MDX_bleed_2" )
		self.label_MDX_bleed_2.Wrap( -1 )

		fgSizer911.Add( self.label_MDX_bleed_2, 0, wx.TOP, 5 )

		bleed_3Choices = []
		self.bleed_3 = wx.ComboBox( sbSizer21.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 220,-1 ), bleed_3Choices, 0 )
		fgSizer911.Add( self.bleed_3, 0, wx.BOTTOM, 5 )

		bleed_4Choices = []
		self.bleed_4 = wx.ComboBox( sbSizer21.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 220,-1 ), bleed_4Choices, 0 )
		fgSizer911.Add( self.bleed_4, 0, wx.LEFT|wx.RIGHT, 10 )

		self.icon4 = wx.StaticBitmap( sbSizer21.GetStaticBox(), wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, 0 )
		fgSizer911.Add( self.icon4, 0, wx.TOP, 3 )

		self.label_MDX_bleed_3 = wx.StaticText( sbSizer21.GetStaticBox(), wx.ID_ANY, _(u"Remove Music"), wx.DefaultPosition, wx.Size( 150,-1 ), 0, u"MDX_bleed_3" )
		self.label_MDX_bleed_3.Wrap( -1 )

		fgSizer911.Add( self.label_MDX_bleed_3, 0, wx.TOP, 5 )

		bleed_5Choices = []
		self.bleed_5 = wx.ComboBox( sbSizer21.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 220,-1 ), bleed_5Choices, 0 )
		fgSizer911.Add( self.bleed_5, 0, 0, 5 )

		bleed_6Choices = []
		self.bleed_6 = wx.ComboBox( sbSizer21.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 220,-1 ), bleed_6Choices, 0 )
		fgSizer911.Add( self.bleed_6, 0, wx.LEFT|wx.RIGHT, 10 )

		self.icon5 = wx.StaticBitmap( sbSizer21.GetStaticBox(), wx.ID_ANY, wx.NullBitmap, wx.DefaultPosition, wx.DefaultSize, 0 )
		fgSizer911.Add( self.icon5, 0, wx.TOP, 3 )


		sbSizer21.Add( fgSizer911, 0, wx.BOTTOM|wx.EXPAND|wx.LEFT, 5 )


		box1.Add( sbSizer21, 0, wx.BOTTOM|wx.EXPAND|wx.LEFT|wx.RIGHT, 5 )

		sbSizer3 = wx.StaticBoxSizer( wx.StaticBox( self.Tab_Settings, wx.ID_ANY, _(u"[ Options ]") ), wx.VERTICAL )

		fgSizer8 = wx.FlexGridSizer( 0, 3, 0, 0 )
		fgSizer8.SetFlexibleDirection( wx.BOTH )
		fgSizer8.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )

		self.label_speed = wx.StaticText( sbSizer3.GetStaticBox(), wx.ID_ANY, _(u"Speed"), wx.DefaultPosition, wx.Size( 150,-1 ), 0, u"speed" )
		self.label_speed.Wrap( -1 )

		fgSizer8.Add( self.label_speed, 0, wx.TOP, 5 )

		self.speed = wx.Slider( sbSizer3.GetStaticBox(), wx.ID_ANY, 2, 0, 4, wx.DefaultPosition, wx.Size( 345,-1 ), wx.SL_HORIZONTAL )
		fgSizer8.Add( self.speed, 0, wx.ALL, 2 )

		self.speed_readout = wx.StaticText( sbSizer3.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTER_HORIZONTAL )
		self.speed_readout.Wrap( -1 )

		self.speed_readout.SetMinSize( wx.Size( 100,-1 ) )

		fgSizer8.Add( self.speed_readout, 0, wx.TOP, 3 )

		self.label_chunks = wx.StaticText( sbSizer3.GetStaticBox(), wx.ID_ANY, _(u"Chunk Size"), wx.DefaultPosition, wx.Size( 150,-1 ), 0, u"chunks" )
		self.label_chunks.Wrap( -1 )

		fgSizer8.Add( self.label_chunks, 0, wx.TOP, 5 )

		self.chunk_size = wx.Slider( sbSizer3.GetStaticBox(), wx.ID_ANY, 5, 1, 10, wx.DefaultPosition, wx.Size( 345,-1 ), wx.SL_HORIZONTAL )
		fgSizer8.Add( self.chunk_size, 0, wx.ALL, 2 )

		self.chunk_size_readout = wx.StaticText( sbSizer3.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.ALIGN_CENTER_HORIZONTAL )
		self.chunk_size_readout.Wrap( -1 )

		self.chunk_size_readout.SetMinSize( wx.Size( 100,-1 ) )

		fgSizer8.Add( self.chunk_size_readout, 0, wx.TOP, 3 )


		sbSizer3.Add( fgSizer8, 0, wx.BOTTOM|wx.EXPAND|wx.LEFT, 5 )


		box1.Add( sbSizer3, 0, wx.BOTTOM|wx.EXPAND|wx.LEFT|wx.RIGHT, 5 )

		sbSizer4 = wx.StaticBoxSizer( wx.StaticBox( self.Tab_Settings, wx.ID_ANY, _(u"[ BONUS ]") ), wx.VERTICAL )

		fgSizer9 = wx.FlexGridSizer( 0, 4, 0, 20 )
		fgSizer9.SetFlexibleDirection( wx.BOTH )
		fgSizer9.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )

		self.label_debug = wx.StaticText( sbSizer4.GetStaticBox(), wx.ID_ANY, _(u"DEBUG Mode"), wx.DefaultPosition, wx.Size( 130,-1 ), 0, u"debug" )
		self.label_debug.Wrap( -1 )

		fgSizer9.Add( self.label_debug, 0, wx.TOP, 5 )

		self.DEBUG = wx.CheckBox( sbSizer4.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		fgSizer9.Add( self.DEBUG, 0, wx.TOP, 8 )

		self.label_kill_end = wx.StaticText( sbSizer4.GetStaticBox(), wx.ID_ANY, _(u"This is the END ..."), wx.DefaultPosition, wx.Size( 130,-1 ), 0, u"kill_end" )
		self.label_kill_end.Wrap( -1 )

		fgSizer9.Add( self.label_kill_end, 0, wx.TOP, 5 )

		self.KILL_on_END = wx.CheckBox( sbSizer4.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( -1,-1 ), 0 )
		fgSizer9.Add( self.KILL_on_END, 0, wx.TOP, 8 )

		self.label_god_mode = wx.StaticText( sbSizer4.GetStaticBox(), wx.ID_ANY, _(u"GOD Mode"), wx.DefaultPosition, wx.Size( 130,-1 ), 0, u"god_mode" )
		self.label_god_mode.Wrap( -1 )

		fgSizer9.Add( self.label_god_mode, 0, wx.TOP, 5 )

		self.GOD_MODE = wx.CheckBox( sbSizer4.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		fgSizer9.Add( self.GOD_MODE, 0, wx.TOP, 8 )


		sbSizer4.Add( fgSizer9, 0, wx.BOTTOM|wx.EXPAND|wx.LEFT, 5 )


		box1.Add( sbSizer4, 0, wx.BOTTOM|wx.EXPAND|wx.LEFT|wx.RIGHT, 5 )

		wSizer1 = wx.WrapSizer( wx.HORIZONTAL, wx.WRAPSIZER_DEFAULT_FLAGS )

		self.Btn_Preset_1 = wx.Button( self.Tab_Settings, wx.ID_ANY, _(u"1"), wx.DefaultPosition, wx.Size( 32,32 ), 0 )
		self.Btn_Preset_1.SetForegroundColour( wx.Colour( 255, 255, 255 ) )
		self.Btn_Preset_1.SetBackgroundColour( wx.Colour( 13, 138, 240 ) )
		self.Btn_Preset_1.SetToolTip( _(u"Preset 1") )

		wSizer1.Add( self.Btn_Preset_1, 0, wx.ALL, 5 )

		self.Btn_Preset_2 = wx.Button( self.Tab_Settings, wx.ID_ANY, _(u"2"), wx.DefaultPosition, wx.Size( 32,32 ), 0 )
		self.Btn_Preset_2.SetForegroundColour( wx.Colour( 255, 255, 255 ) )
		self.Btn_Preset_2.SetBackgroundColour( wx.Colour( 13, 138, 240 ) )
		self.Btn_Preset_2.SetToolTip( _(u"Preset 2") )

		wSizer1.Add( self.Btn_Preset_2, 0, wx.ALL, 5 )

		self.Btn_Preset_3 = wx.Button( self.Tab_Settings, wx.ID_ANY, _(u"3"), wx.DefaultPosition, wx.Size( 32,32 ), 0 )
		self.Btn_Preset_3.SetForegroundColour( wx.Colour( 255, 255, 255 ) )
		self.Btn_Preset_3.SetBackgroundColour( wx.Colour( 13, 138, 240 ) )
		self.Btn_Preset_3.SetToolTip( _(u"Preset 3") )

		wSizer1.Add( self.Btn_Preset_3, 0, wx.ALL, 5 )

		self.Btn_Preset_4 = wx.Button( self.Tab_Settings, wx.ID_ANY, _(u"4"), wx.DefaultPosition, wx.Size( 32,32 ), 0 )
		self.Btn_Preset_4.SetForegroundColour( wx.Colour( 255, 255, 255 ) )
		self.Btn_Preset_4.SetBackgroundColour( wx.Colour( 13, 138, 240 ) )
		self.Btn_Preset_4.SetToolTip( _(u"Preset 4") )

		wSizer1.Add( self.Btn_Preset_4, 0, wx.ALL, 5 )

		self.Btn_Preset_5 = wx.Button( self.Tab_Settings, wx.ID_ANY, _(u"5"), wx.DefaultPosition, wx.Size( 32,32 ), 0 )
		self.Btn_Preset_5.SetForegroundColour( wx.Colour( 255, 255, 255 ) )
		self.Btn_Preset_5.SetBackgroundColour( wx.Colour( 13, 138, 240 ) )
		self.Btn_Preset_5.SetToolTip( _(u"Preset 5") )

		wSizer1.Add( self.Btn_Preset_5, 0, wx.ALL, 5 )


		wSizer1.Add( ( 130, 0), 1, 0, 5 )

		self.Btn_Start = wx.Button( self.Tab_Settings, wx.ID_ANY, _(u"Start"), wx.DefaultPosition, wx.Size( 150,32 ), 0 )

		self.Btn_Start.SetDefault()
		self.Btn_Start.SetForegroundColour( wx.Colour( 255, 255, 255 ) )
		self.Btn_Start.SetBackgroundColour( wx.Colour( 13, 138, 240 ) )

		wSizer1.Add( self.Btn_Start, 0, wx.ALL, 5 )


		box1.Add( wSizer1, 0, wx.EXPAND, 5 )

		self.HELP = wx.html.HtmlWindow( self.Tab_Settings, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.html.HW_SCROLLBAR_NEVER|wx.BORDER_SIMPLE )
		self.HELP.SetBackgroundColour( wx.Colour( 255, 255, 210 ) )

		box1.Add( self.HELP, 10, wx.EXPAND|wx.TOP, 5 )


		self.Tab_Settings.SetSizer( box1 )
		self.Tab_Settings.Layout()
		box1.Fit( self.Tab_Settings )
		self.Tabs.AddPage( self.Tab_Settings, _(u" Settings  "), True )
		self.Tab_Progress = wx.Panel( self.Tabs, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		box2 = wx.BoxSizer( wx.VERTICAL )

		bSizer61 = wx.BoxSizer( wx.HORIZONTAL )

		self.GPU = wx.StaticText( self.Tab_Progress, wx.ID_ANY, _(u"Using GPU (0 GB) ►"), wx.DefaultPosition, wx.Size( 200,-1 ), 0 )
		self.GPU.Wrap( -1 )

		self.GPU.SetFont( wx.Font( 12, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_BOLD, False, "Tahoma" ) )

		bSizer61.Add( self.GPU, 0, wx.BOTTOM|wx.LEFT|wx.TOP, 11 )

		self.GPU_info = wx.StaticText( self.Tab_Progress, wx.ID_ANY, _(u"0 GB - 0 %"), wx.DefaultPosition, wx.Size( -1,-1 ), 0 )
		self.GPU_info.Wrap( -1 )

		bSizer61.Add( self.GPU_info, 0, wx.BOTTOM|wx.EXPAND|wx.TOP, 11 )


		box2.Add( bSizer61, 0, wx.ALL|wx.EXPAND, 5 )

		self.CONSOLE = wx.html.HtmlWindow( self.Tab_Progress, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.html.HW_SCROLLBAR_AUTO )
		box2.Add( self.CONSOLE, 1, wx.EXPAND, 5 )

		bSizer6 = wx.BoxSizer( wx.HORIZONTAL )

		self.Progress_Bar = wx.Gauge( self.Tab_Progress, wx.ID_ANY, 100, wx.DefaultPosition, wx.Size( 320,16 ), wx.GA_HORIZONTAL|wx.GA_SMOOTH )
		self.Progress_Bar.SetValue( 0 )
		bSizer6.Add( self.Progress_Bar, 0, wx.EXPAND|wx.RIGHT, 15 )

		self.Progress_Text = wx.StaticText( self.Tab_Progress, wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.Size( 350,-1 ), 0 )
		self.Progress_Text.Wrap( -1 )

		self.Progress_Text.SetFont( wx.Font( 11, wx.FONTFAMILY_SWISS, wx.FONTSTYLE_NORMAL, wx.FONTWEIGHT_NORMAL, False, "Tahoma" ) )

		bSizer6.Add( self.Progress_Text, 0, 0, 5 )


		box2.Add( bSizer6, 0, wx.ALL|wx.EXPAND, 5 )


		self.Tab_Progress.SetSizer( box2 )
		self.Tab_Progress.Layout()
		box2.Fit( self.Tab_Progress )
		self.Tabs.AddPage( self.Tab_Progress, _(u" Progress  "), False )
		self.Tab_System = wx.Panel( self.Tabs, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		box3 = wx.BoxSizer( wx.VERTICAL )

		self.Btn_SysInfo = wx.Button( self.Tab_System, wx.ID_ANY, _(u"Get System informations"), wx.DefaultPosition, wx.Size( 240,32 ), 0 )
		self.Btn_SysInfo.SetForegroundColour( wx.Colour( 255, 255, 255 ) )
		self.Btn_SysInfo.SetBackgroundColour( wx.Colour( 13, 138, 240 ) )

		box3.Add( self.Btn_SysInfo, 0, wx.ALL, 5 )

		self.sys_info = wx.html.HtmlWindow( self.Tab_System, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.html.HW_SCROLLBAR_AUTO )
		box3.Add( self.sys_info, 1, wx.EXPAND, 5 )


		self.Tab_System.SetSizer( box3 )
		self.Tab_System.Layout()
		box3.Fit( self.Tab_System )
		self.Tabs.AddPage( self.Tab_System, _(u" System Info  "), False )

		MainSizer.Add( self.Tabs, 1, wx.ALL|wx.EXPAND, 5 )


		self.SetSizer( MainSizer )
		self.Layout()

		self.Centre( wx.BOTH )

		# Connect Events
		self.Bind( wx.EVT_CLOSE, self.Form_OnClose )
		self.Tabs.Bind( wx.EVT_NOTEBOOK_PAGE_CHANGED, self.Tab_Changed )
		self.label_input.Bind( wx.EVT_ENTER_WINDOW, self.Show_Help )
		self.Btn_input_Path.Bind( wx.EVT_BUTTON, self.Btn_input_Path_OnClick )
		self.Btn_input_File.Bind( wx.EVT_BUTTON, self.Btn_input_File_OnClick )
		self.input_path.Bind( wx.EVT_TEXT, self.input_path_OnChange )
		self.label_normalize.Bind( wx.EVT_ENTER_WINDOW, self.Show_Help )
		self.normalize.Bind( wx.EVT_KEY_DOWN, self.ComboBox_OnKeyDown )
		self.label_output.Bind( wx.EVT_ENTER_WINDOW, self.Show_Help )
		self.Btn_output_Path.Bind( wx.EVT_BUTTON, self.Btn_output_Path_OnClick )
		self.output_path.Bind( wx.EVT_TEXT, self.output_path_OnChange )
		self.label_format.Bind( wx.EVT_ENTER_WINDOW, self.Show_Help )
		self.output_format.Bind( wx.EVT_KEY_DOWN, self.ComboBox_OnKeyDown )
		self.label_silent.Bind( wx.EVT_ENTER_WINDOW, self.Show_Help )
		self.silent.Bind( wx.EVT_KEY_DOWN, self.ComboBox_OnKeyDown )
		self.label_infra_bass.Bind( wx.EVT_ENTER_WINDOW, self.Show_Help )
		self.label_MDX_music.Bind( wx.EVT_ENTER_WINDOW, self.Show_Help )
		self.music_1.Bind( wx.EVT_KEY_DOWN, self.ComboBox_OnKeyDown )
		self.music_2.Bind( wx.EVT_KEY_DOWN, self.ComboBox_OnKeyDown )
		self.label_MDX_vocal.Bind( wx.EVT_ENTER_WINDOW, self.Show_Help )
		self.vocal_1.Bind( wx.EVT_KEY_DOWN, self.ComboBox_OnKeyDown )
		self.vocal_2.Bind( wx.EVT_KEY_DOWN, self.ComboBox_OnKeyDown )
		self.label_MDX_bleed_1.Bind( wx.EVT_ENTER_WINDOW, self.Show_Help )
		self.bleed_1.Bind( wx.EVT_KEY_DOWN, self.ComboBox_OnKeyDown )
		self.bleed_2.Bind( wx.EVT_KEY_DOWN, self.ComboBox_OnKeyDown )
		self.label_vocal_pass.Bind( wx.EVT_ENTER_WINDOW, self.Show_Help )
		self.high_pass.Bind( wx.EVT_SLIDER, self.high_pass_OnSlider )
		self.low_pass.Bind( wx.EVT_SLIDER, self.low_pass_OnSlider )
		self.label_MDX_bleed_2.Bind( wx.EVT_ENTER_WINDOW, self.Show_Help )
		self.bleed_3.Bind( wx.EVT_KEY_DOWN, self.ComboBox_OnKeyDown )
		self.bleed_4.Bind( wx.EVT_KEY_DOWN, self.ComboBox_OnKeyDown )
		self.label_MDX_bleed_3.Bind( wx.EVT_ENTER_WINDOW, self.Show_Help )
		self.bleed_5.Bind( wx.EVT_KEY_DOWN, self.ComboBox_OnKeyDown )
		self.bleed_6.Bind( wx.EVT_KEY_DOWN, self.ComboBox_OnKeyDown )
		self.label_speed.Bind( wx.EVT_ENTER_WINDOW, self.Show_Help )
		self.speed.Bind( wx.EVT_SLIDER, self.speed_OnSlider )
		self.label_chunks.Bind( wx.EVT_ENTER_WINDOW, self.Show_Help )
		self.chunk_size.Bind( wx.EVT_SLIDER, self.chunk_size_OnSlider )
		self.label_debug.Bind( wx.EVT_ENTER_WINDOW, self.Show_Help )
		self.label_kill_end.Bind( wx.EVT_ENTER_WINDOW, self.Show_Help )
		self.label_god_mode.Bind( wx.EVT_ENTER_WINDOW, self.Show_Help )
		self.Btn_Preset_1.Bind( wx.EVT_BUTTON, self.Btn_Preset_1_OnClick )
		self.Btn_Preset_2.Bind( wx.EVT_BUTTON, self.Btn_Preset_2_OnClick )
		self.Btn_Preset_3.Bind( wx.EVT_BUTTON, self.Btn_Preset_3_OnClick )
		self.Btn_Preset_4.Bind( wx.EVT_BUTTON, self.Btn_Preset_4_OnClick )
		self.Btn_Preset_5.Bind( wx.EVT_BUTTON, self.Btn_Preset_5_OnClick )
		self.Btn_Start.Bind( wx.EVT_BUTTON, self.Btn_Start_OnClick )
		self.Btn_SysInfo.Bind( wx.EVT_BUTTON, self.Btn_SysInfo_OnClick )

	def __del__( self ):
		pass


	# Virtual event handlers, override them in your derived class
	def Form_OnClose( self, event ):
		pass

	def Tab_Changed( self, event ):
		pass

	def Show_Help( self, event ):
		pass

	def Btn_input_Path_OnClick( self, event ):
		pass

	def Btn_input_File_OnClick( self, event ):
		pass

	def input_path_OnChange( self, event ):
		pass


	def ComboBox_OnKeyDown( self, event ):
		pass


	def Btn_output_Path_OnClick( self, event ):
		pass

	def output_path_OnChange( self, event ):
		pass
















	def high_pass_OnSlider( self, event ):
		pass

	def low_pass_OnSlider( self, event ):
		pass








	def speed_OnSlider( self, event ):
		pass


	def chunk_size_OnSlider( self, event ):
		pass




	def Btn_Preset_1_OnClick( self, event ):
		pass

	def Btn_Preset_2_OnClick( self, event ):
		pass

	def Btn_Preset_3_OnClick( self, event ):
		pass

	def Btn_Preset_4_OnClick( self, event ):
		pass

	def Btn_Preset_5_OnClick( self, event ):
		pass

	def Btn_Start_OnClick( self, event ):
		pass

	def Btn_SysInfo_OnClick( self, event ):
		pass


