#!python3.10

#   MIT License
#
#   Copyright (c) 2023 - ZFTurbo - Start the project MVSEP-MDX23 (music separation model)
#   Copyright (c) 2023 - Jarredou - Did all the job for Inference !!
#   Copyright (c) 2023 - Captain FLAM - Heavily modified ! (GUI, sequential processing, etc ...)
#
#   https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model
#   https://github.com/jarredou/MVSEP-MDX23-Colab_v2/
#   https://github.com/Captain-FLAM/KaraFan


import os, gc, sys, csv, argparse, requests, io, base64
import regex as re
import numpy as np
import onnxruntime as ort
import torch, torch.nn as nn
from time import time

import librosa, soundfile as sf
from pydub import AudioSegment

# ONLY for MDX23C models
#  import yaml
#  from ml_collections import ConfigDict

import ipywidgets as widgets
from IPython.display import display, HTML

# from tqdm.auto import tqdm  # Auto : Progress Bar in GUI with ipywidgets
# from tqdm.contrib import DummyTqdmFile

import App.settings, App.audio_utils, App.compare

# from App.tfc_tdf_v3 import TFC_TDF_net

isColab = False
KILL_on_END = False

class Conv_TDF_net_trim_model(nn.Module):

	def __init__(self, device, target_stem, neuron_blocks, model_params, hop=1024):

		super(Conv_TDF_net_trim_model, self).__init__()
		
		self.dim_c = 4
		self.dim_f = model_params['dim_F_set']
		self.dim_t = 2 ** model_params['dim_T_set']
		self.n_fft = model_params['N_FFT_scale']
		self.hop = hop
		self.n_bins = self.n_fft // 2 + 1
		self.chunk_size = hop * (self.dim_t - 1)
		self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(device)
		self.target_stem = target_stem

		out_c = self.dim_c * 4 if target_stem == '*' else self.dim_c
		self.freq_pad = torch.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t]).to(device)
		
  		# Only used by "forward()" method
		# self.n = neuron_blocks // 2

	def stft(self, x):
		x = x.reshape([-1, self.chunk_size])
		x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True, return_complex=True)
		x = torch.view_as_real(x)
		x = x.permute([0, 3, 1, 2])
		x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, self.dim_c, self.n_bins, self.dim_t])
		return x[:, :, :self.dim_f]

	def istft(self, x, freq_pad=None):
		freq_pad = self.freq_pad.repeat([x.shape[0], 1, 1, 1]) if freq_pad is None else freq_pad
		x = torch.cat([x, freq_pad], -2)
		x = x.reshape([-1, 2, 2, self.n_bins, self.dim_t]).reshape([-1, 2, self.n_bins, self.dim_t])
		x = x.permute([0, 2, 3, 1])
		x = x.contiguous()
		x = torch.view_as_complex(x)
		x = torch.istft(x, n_fft=self.n_fft, hop_length=self.hop, window=self.window, center=True)
		return x.reshape([-1, 2, self.chunk_size])

	# Not used : only for training Models !
	#
	# def forward(self, x):
	# 	x = self.first_conv(x)
	# 	x = x.transpose(-1, -2)
	#
	# 	ds_outputs = []
	# 	for i in range(self.n):
	# 		x = self.ds_dense[i](x)
	# 		ds_outputs.append(x)
	# 		x = self.ds[i](x)
	#
	# 	x = self.mid_dense(x)
	# 	for i in range(self.n):
	# 		x = self.us[i](x)
	# 		x *= ds_outputs[-i - 1]
	# 		x = self.us_dense[i](x)
	#
	# 	x = x.transpose(-1, -2)
	# 	x = self.final_conv(x)
	# 	return x

def get_models(device, model_params, stem):
	# ??? NOT so simple ... ???
	# FFT = 7680  --> Narrow Band
	# FFT = 6144  --> FULL Band
	model = Conv_TDF_net_trim_model(
		device,
		# I suppose you can use '*' to get both vocals and instrum, with the new MDX23C model ...
		'vocals' if stem == 'Vocals' else 'instrum',
		11,
		model_params
	)
	return [model]

# def demix_base_mdxv3(config, model, mix, device, overlap):
# 	mix = torch.tensor(mix, dtype=torch.float32)
# 	try:
# 		S = model.num_target_instruments
# 	except Exception as e:
# 		S = model.module.num_target_instruments

# 	mdx_window_size = config.inference.dim_t
	
# 	# batch_size = config.inference.batch_size
# 	batch_size = 1
# 	C = config.audio.hop_length * (mdx_window_size - 1)
	
# 	H = C // overlap
# 	L = mix.shape[1]
# 	pad_size = H - (L - C) % H
# 	mix = torch.cat([torch.zeros(2, C - H), mix, torch.zeros(2, pad_size + C - H)], 1)
# 	mix = mix.to(device)

# 	chunks = []
# 	i = 0
# 	while i + C <= mix.shape[1]:
# 		chunks.append(mix[:, i:i + C])
# 		i += H
# 	chunks = torch.stack(chunks)

# 	batches = []
# 	i = 0
# 	while i < len(chunks):
# 		batches.append(chunks[i:i + batch_size])
# 		i = i + batch_size

# 	X = torch.zeros(S, 2, C - H) if S > 1 else torch.zeros(2, C - H)
# 	X = X.to(device)

# 	with torch.cuda.amp.autocast():
# 		with torch.no_grad():
# 			for batch in tqdm(batches, ncols=60):
# 				# self.running_inference_progress_bar(len(batches))
# 				x = model(batch)
# 				for w in x:
# 					a = X[..., :-(C - H)]
# 					b = X[..., -(C - H):] + w[..., :(C - H)]
# 					c = w[..., (C - H):]
# 					X = torch.cat([a, b, c], -1)

# 	estimated_sources = X[..., C - H:-(pad_size + C - H)] / overlap

# 	if S > 1:
# 		return {k: v for k, v in zip(config.training.instruments, estimated_sources.cpu().numpy())}
	
# 	est_s = estimated_sources.cpu().numpy()
# 	return est_s

# def demix_full_mdx23c(mix, device, overlap):
# 	model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "Models")

# 	remote_url_mdxv3 = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/MDX23C_D1581.ckpt'
# 	remote_url_conf = 'https://raw.githubusercontent.com/Anjok07/ultimatevocalremovergui/new-patch-3-20/models/MDX_Net_Models/model_data/mdx_c_configs/model_2_stem_061321.yaml'
# 	if not os.path.isfile(os.path.join(model_folder, 'MDX23C_D1581.ckpt')):
# 		torch.hub.download_url_to_file(remote_url_mdxv3, os.path.join(model_folder, 'MDX23C_D1581.ckpt'))
# 	if not os.path.isfile(os.path.join(model_folder, 'model_2_stem_061321.yaml')):
# 		torch.hub.download_url_to_file(remote_url_conf, os.path.join(model_folder, 'model_2_stem_061321.yaml'))

# 	with open(os.path.join(model_folder, 'model_2_stem_061321.yaml')) as f:
# 		config = ConfigDict(yaml.load(f, Loader=yaml.FullLoader))

# 	model = TFC_TDF_net(config)
# 	model.load_state_dict(torch.load(os.path.join(model_folder, 'MDX23C_D1581.ckpt')))
# 	device = torch.device(device)
# 	model = model.to(device)
# 	model.eval()

# 	sources = demix_base_mdxv3(config, model, mix, device, overlap)
# 	del model
# 	gc.collect()

# 	return sources

def demix_base(mix, device, models, infer_session):
	sources = []
	n_sample = mix.shape[1]
	for model in models:
		trim = model.n_fft // 2
		gen_size = model.chunk_size - 2 * trim
		pad = gen_size - n_sample % gen_size
		mix_p = np.concatenate(
			(
				np.zeros((2, trim)),
				mix,
				np.zeros((2, pad)),
				np.zeros((2, trim))
			), 1
		)

		mix_waves = []
		i = 0
		while i < n_sample + pad:
			waves = np.array(mix_p[:, i:i + model.chunk_size])
			mix_waves.append(waves)
			i += gen_size
		mix_waves = np.array(mix_waves)
		mix_waves = torch.tensor(mix_waves, dtype=torch.float32).to(device)

		try:
			with torch.no_grad():
				_ort = infer_session
				stft_res = model.stft(mix_waves)
				res = _ort.run(None, {'input': stft_res.cpu().numpy()})[0]
				ten = torch.tensor(res)
				tar_waves = model.istft(ten.to(device))
				tar_waves = tar_waves.cpu()
				tar_signal = tar_waves[:, :, trim:-trim].transpose(0, 1).reshape(2, -1).numpy()[:, :-pad]

			sources.append(tar_signal)

		except Exception as e:
			print("\n\nError in demix_base() with Torch : ", e)
			Exit_Notebook()
	
	return np.array(sources)


class MusicSeparationModel:

	def __init__(self, params, config):

		self.Gdrive   = params['Gdrive']
		self.CONSOLE  = params['CONSOLE']
		self.Progress = params['Progress']

		self.output_format		= config['PROCESS']['output_format']
#		self.preset_genre		= config['PROCESS']['preset_genre']
		self.normalize			= (config['PROCESS']['normalize'].lower() == "true")
		self.REPAIR_MUSIC		= config['PROCESS']['REPAIR_MUSIC']
#		self.overlap_MDXv3		= int(config['OPTIONS']['overlap_MDXv3'])
		self.chunk_size			= int(config['OPTIONS']['chunk_size'])
		self.PREVIEWS			= (config['BONUS']['PREVIEWS'].lower() == "true")
		self.DEBUG				= (config['BONUS']['DEBUG'].lower() == "true")
		self.GOD_MODE			= (config['BONUS']['GOD_MODE'].lower() == "true")
		self.TEST_MODE			= (config['BONUS']['TEST_MODE'].lower() == "true")
		self.large_gpu			= (config['BONUS']['large_gpu'].lower() == "true")

		self.output = os.path.join(self.Gdrive, config['PATHS']['output'])
		
		self.device = 'cpu'
		if torch.cuda.is_available():  self.device = 'cuda:0'
		
		if self.device == 'cpu':
			print('<div style="font-size:18px;font-weight:bold;color:#ff0040;">Warning ! CPU is used instead of GPU for processing.<br>Processing will be very slow !!</div>')
		else:
			print('<div style="font-size:18px;font-weight:bold;color:#00b32d;">It\'s OK -> GPU is used for processing !!</div>')
		
		if self.device == 'cpu':
			self.providers = ["CPUExecutionProvider"]
		else:
			self.providers = ["CUDAExecutionProvider"]

		# Set BigShifts & Volume Compensation from Quality option
		match config['OPTIONS']['speed']:
			case 'Fastest':
				Quality = "x0"
				self.shifts_vocals	=  1
				self.shifts_instru	=  1
				self.shifts_SRS		=  1
				self.shifts_filter	=  1
			case 'Fast':
				Quality = "x1"
				self.shifts_vocals	=  6
				self.shifts_instru	=  6
				self.shifts_SRS		=  2
				self.shifts_filter	=  2
			case 'Medium':
				Quality = "x2"
				self.shifts_vocals	= 12
				self.shifts_instru	= 12
				self.shifts_SRS		=  4
				self.shifts_filter	=  4
			case 'Slow':
				Quality = "x3"
				self.shifts_vocals	= 16
				self.shifts_instru	= 16
				self.shifts_SRS		=  8
				self.shifts_filter	=  8
			case 'Slowest':
				Quality = "x4"
				self.shifts_vocals	= 21
				self.shifts_instru	= 21
				self.shifts_SRS		= 12
				self.shifts_filter	= 12
		
		self.Best_Compensations = []
		self.Compensation_Vocal_ENS = 1.0
		self.Compensation_Music_SUB = 1.0
		self.Compensation_Music_ENS = 1.0

		# MDX-B models initialization

		self.models = { 'vocal': [], 'music': [], 'filter': [] }
		self.MDX = {}

		# Load Models parameters
		with open(os.path.join(params['Project'], "App", "Models_DATA.csv")) as csvfile:
			reader = csv.DictReader(csvfile, quoting=csv.QUOTE_ALL)
			for row in reader:
				# ignore "Other" stems for now !
				name = row['Name']
				if name == "":  continue

				# Set Volume Compensation from Quality option for "Music_SUB"
				#  ->  MANDATORY to be set in CSV !!
				match name:
					case "VOCAL_ENS_x2":
						if len(self.models['vocal']) == 2:	self.Compensation_Vocal_ENS = float(row['Comp_' + Quality])
					case "VOCAL_ENS_x3":
						if len(self.models['vocal']) > 2:	self.Compensation_Vocal_ENS = float(row['Comp_' + Quality])
					case "MUSIC_ENS_x2":
						if len(self.models['music']) == 2:	self.Compensation_Music_ENS = float(row['Comp_' + Quality])
					# 2 VOCALS models !!
					case "MUSIC_SUB_x2":
						if len(self.models['vocal']) == 2:	self.Compensation_Music_SUB = float(row['Comp_' + Quality])
					# 3 VOCALS models !!
					case "MUSIC_SUB_x3":
						if len(self.models['vocal']) > 2:	self.Compensation_Music_SUB = float(row['Comp_' + Quality])
					case _:
						if name == config['PROCESS']['vocals_1'] or name == config['PROCESS']['vocals_2'] \
						or name == config['PROCESS']['vocals_3'] or name == config['PROCESS']['vocals_4']:
							self.models['vocal'].append(row)
						elif name == config['PROCESS']['instru_1'] or name == config['PROCESS']['instru_2']:
							self.models['music'].append(row)
						
						# Special case for "Filters" : can be Vocal or Instrumental !
						if name == config['PROCESS']['filter_1'] or name == config['PROCESS']['filter_2'] \
						or name == config['PROCESS']['filter_3'] or name == config['PROCESS']['filter_4']:
							self.models['filter'].append(row)

		# Download Models to :
		models_path	= os.path.join(self.Gdrive, "KaraFan_user", "Models")

		for stem in self.models:
			for model in self.models[stem]:
				model['Cut_OFF']		= int(model['Cut_OFF'])
				model['N_FFT_scale']	= int(model['N_FFT_scale'])
				model['dim_F_set']		= int(model['dim_F_set'])
				model['dim_T_set']		= int(model['dim_T_set'])

				# IMPORTANT : Volume Compensations are specific for each model !!!
				# Empirical values to get the best SDR !
				# TODO : Need to be checked against each models combinations !!

				if model['Comp_' + Quality] == "":
					model['Compensation'] = 1.0
				else:
					model['Compensation'] = float(model['Comp_' + Quality])
				
				model['PATH'] = Download_Model(model, models_path, self.CONSOLE, self.Progress)
		
		# Load Models
		if self.large_gpu: 
			print("Large GPU mode is enabled : Loading models now...")

			for stem in self.models:
				for model in self.models[stem]:  self.Load_MDX(model)
	
		# In case of changes, don't forget to update the function in GUI !!
		# - on_Del_Vocals_clicked()
		# - on_Del_Music_clicked()
		self.AudioFiles = [
			"NORMALIZED",
			"Vocal extract",
			"Music extract",
			"Bleedings in Music",
			"Vocal FINAL",
			"Music FINAL",
		]
		self.AudioFiles_Mandatory = [4, 5]  # Vocal & Music FINAL
		self.AudioFiles_Debug = [0, 1, 2]  # NORMALIZED, Vocal extract, Music extract
		
		# DEBUG : Reload "Bleedings in Music" files with GOD MODE
		self.AudioFiles_Debug.append(3)
		
	# ******************************************************************
	# ****    This is the MAGIC RECIPE , the heart of KaraFan !!    ****
	# ******************************************************************

	def SEPARATE(self, file, BATCH_MODE):

		name = os.path.splitext(os.path.basename(file))[0]
		
		#*************************************************
		#****  DEBUG  ->  TESTING SDR for DEVELOPERS  ****
		#*************************************************
		
		# Put some "song_XXX.flac" from "Gdrive > KaraFan_user > Multi-Song" in your "Music" folder
		# That's all !!
		# (only the song file, not "instrum.flac" or "vocals.flac" from "Stems" folder)

		self.SDR_Testing = (name.startswith("song_") and re.match(r"^song_\d{3}$", name) is not None)

		#*************************************************

		start_time = time()

		self.BATCH_MODE = BATCH_MODE
		if self.CONSOLE:	print("Go with : <b>" + name + "</b>")
		else:				print("Go with : " + name)

		# Create a folder based on input audio file's name
		self.song_output_path = os.path.join(self.output, name)
		if not os.path.exists(self.song_output_path): os.makedirs(self.song_output_path)
		
		# TODO : sr = None --> uses the native sampling rate (if 48 Khz or 96 Khz), maybe not good for MDX models ??
		original_audio, self.sample_rate = librosa.load(file, mono=False, sr = 44100)  # Resample to 44.1 Khz
		
		# Convert mono to stereo (if needed)
		if len(original_audio.shape) == 1:
			original_audio = np.stack([original_audio, original_audio], axis=0)
		
		# TODO : Get the cut-off frequency of the input audio
		# self.original_cutoff = App.audio_utils.Find_Cut_OFF(original_audio, self.sample_rate)
		
		self.original_cutoff = self.sample_rate // 2
		
		print(f"Input Audio : {original_audio.shape} - Rate : {self.sample_rate} Hz / Cut-OFF : {self.original_cutoff} Hz")
		
		# ****  START PROCESSING  ****

		if self.normalize:
			normalized = self.Check_Already_Processed(0)

			if normalized is None:
				print("► Normalizing audio")
				normalized = App.audio_utils.Normalize(original_audio)

				self.Save_Audio(0, normalized)
		else:
			normalized = original_audio
		
		# print("► Processing vocals with MDX23C model")

		# sources3 = demix_full_mdx23c(normalized, self.device, self.overlap_MDXv3)
		# vocals3 = (match_array_shapes(sources3['Vocals'], normalized) \
		# 		+ Pass_filter('lowpass', 14700, normalized - match_array_shapes(sources3['Instrumental'], normalized), 44100)) / 2
		
		# if self.DEBUG:
		#	self.Save_Audio("Vocal_MDX23C", vocals3)
		
		# 1 - Extract Vocals with MDX models

		vocal_extracts = []
		for model in self.models['vocal']:
			audio = self.Check_Already_Processed(1, model['Name'])
			if audio is None:
				audio = self.Extract_with_Model("Vocal", normalized, model)

				# Apply silence filter
				audio = App.audio_utils.Silent(audio, self.sample_rate)
		
				self.Save_Audio(1, audio, model['Name'])
			
			vocal_extracts.append(audio)
		
		if len(vocal_extracts) > 1:
			print("► Make Ensemble Vocals")
			vocal_ensemble = App.audio_utils.Make_Ensemble('Max', vocal_extracts)  # MAX and not average, because it's Vocals !!

			# DEBUG : Test different values for SDR Volume Compensation
			if self.DEBUG and self.SDR_Testing:
				Best_Volume = App.compare.SDR_Volumes("Vocal", vocal_ensemble, self.Compensation_Vocal_ENS, self.song_output_path, self.Gdrive)

				if self.Compensation_Vocal_ENS != Best_Volume:
					self.Compensation_Vocal_ENS = Best_Volume
					self.Best_Compensations.append('Best Compensation for "Vocal Ensemble" : {:5.3f}'.format(Best_Volume))

			vocal_ensemble = vocal_ensemble * self.Compensation_Vocal_ENS

			# DEBUG : if "pass band" are applied at the end (FINAL SAVE), just to compare ...
			# self.Save_Audio("1 - "+ self.AudioFiles[1] +" - Ensemble", vocal_ensemble)
		else:
			vocal_ensemble = vocal_extracts[0]
		
		del vocal_extracts;  gc.collect()
		
		# TODO
		# test_audio = self.Extract_with_Model("Filter", vocal_ensemble, self.models['music'][0])
		# self.Save_Audio("X - Test Vocal Filter", test_audio)
		# vocal_ensemble = vocal_ensemble - test_audio

		# 2 - Get Music by substracting Vocals from original audio (for instrumental not captured by MDX models)
		
		print("► Get Music by substracting Vocals from original audio")
		music_sub = normalized - vocal_ensemble

		# DEBUG : Test different values for SDR Volume Compensation
		if self.DEBUG and self.SDR_Testing:
			Best_Volume = App.compare.SDR_Volumes("Music", music_sub, self.Compensation_Music_SUB, self.song_output_path, self.Gdrive)

			if self.Compensation_Music_SUB != Best_Volume:
				self.Compensation_Music_SUB = Best_Volume
				self.Best_Compensations.append('Best Compensation for "Music SUB"      : {:5.3f}'.format(Best_Volume))

		music_sub = music_sub * self.Compensation_Music_SUB

		print("► Save Vocals FINAL !")

		# Better SDR ??
		# vocal_ensemble = App.audio_utils.Pass_filter('highpass', 50, vocal_ensemble, self.sample_rate, order = 100)
		# vocal_ensemble = App.audio_utils.Pass_filter('lowpass', 16000, vocal_ensemble, self.sample_rate, order = 4)

		# Apply silence filter : -60 dB !
		vocal_ensemble = App.audio_utils.Silent(vocal_ensemble, self.sample_rate, threshold_db = -60)

		self.Save_Audio(4, vocal_ensemble)

		# 3 - Extract Music with MDX models

		if self.REPAIR_MUSIC != "No !!":
			music_extracts = []
			
			for model in self.models['music']:
				audio = self.Check_Already_Processed(2, model['Name'])
				if audio is None:
					audio = self.Extract_with_Model("Music", normalized, model)

					self.Save_Audio(2, audio, model['Name'])
				
				music_extracts.append(audio)
				
			if len(music_extracts) > 1:
				print("► Make Ensemble Music")
				
				if self.REPAIR_MUSIC == "Average Mix":
					music_ensemble = App.audio_utils.Make_Ensemble('Average', music_extracts)
				elif self.REPAIR_MUSIC == "Maximum Mix":
					music_ensemble = App.audio_utils.Make_Ensemble('Max', music_extracts)
				
				# DEBUG : Test different values for SDR Volume Compensation
				if self.DEBUG and self.SDR_Testing:
					Best_Volume = App.compare.SDR_Volumes("Music", music_ensemble, self.Compensation_Music_ENS, self.song_output_path, self.Gdrive)

					if self.Compensation_Music_ENS != Best_Volume:
						self.Compensation_Music_ENS = Best_Volume
						self.Best_Compensations.append('Best Compensation for "Music Ensemble" : {:5.3f}'.format(Best_Volume))

				music_ensemble = music_ensemble * self.Compensation_Music_ENS

				self.Save_Audio("2 - "+ self.AudioFiles[2] +" - Ensemble", music_ensemble)
			else:
				music_ensemble = music_extracts[0]

			del music_extracts;  gc.collect()

		# 4 - Pass Music through Filters (remove VOCALS bleedings)

		if self.REPAIR_MUSIC != "No !!" and len(self.models['filter']) > 0:
			
			filters = []
			for model in self.models['filter']:
				audio = self.Check_Already_Processed(3, model['Name'])
				if audio is None:
					audio = self.Extract_with_Model("Filter", music_ensemble, model)

					# If model Stem is Music, substraction is needed !
					if model['Stem'] == "Instrumental":  audio = music_ensemble - audio

					# Apply silence filter
					audio = App.audio_utils.Silent(audio, self.sample_rate)

					self.Save_Audio(3, audio, model['Name'])
				
				filters.append(audio)

			if len(filters) > 1:
				print("► Make Ensemble Filters")
				filters_ensemble = App.audio_utils.Make_Ensemble('Max', filters)  # MAX and not average, because it's Vocals !!

				self.Save_Audio("3 - "+ self.AudioFiles[3] +" - Ensemble", filters_ensemble)
			else:
				filters_ensemble = filters[0]

			#  Remove Vocals Bleedings
			music_ensemble = music_ensemble - filters_ensemble
			
			# Used for SDR testing --> Do not add "Bleeedings" in name, else it will be ignored !!
			if self.DEBUG and self.SDR_Testing:
				self.Save_Audio("3 - Music extract SUB - Bleeds", music_ensemble)

			del filters;  gc.collect()

		# 5 - Repair Music

		if self.REPAIR_MUSIC != "No !!":
			if self.DEBUG and self.SDR_Testing:
				self.Save_Audio("2 - Music - SUB", music_sub)

			print("► Repair Music")

			if self.REPAIR_MUSIC == "Average Mix":
				music_final = App.audio_utils.Make_Ensemble('Average', [music_sub, music_ensemble])
			elif self.REPAIR_MUSIC == "Maximum Mix":
				music_final = App.audio_utils.Make_Ensemble('Max', [music_sub, music_ensemble])
			
			# TODO : Strange stuff, need to be checked
			# I don't know yet if it takes the maximum of Music_SUB (lost instruments) ??
			# The 1st one must be ...
			# music_final = np.where(np.abs(music_sub) >= np.abs(music_final), music_sub, music_final)
			# ... but the 2nd one get a BETTER SDR score !!
			# music_final = np.where(np.abs(music_final) >= np.abs(music_sub), music_sub, music_final)
		else:
			music_final = music_sub

		# 6 - FINAL saving

		print("► Save Music FINAL !")
		
		# Apply silence filter : -60 dB !
		music_final = App.audio_utils.Silent(music_final, self.sample_rate, threshold_db = -60)

		self.Save_Audio(5, music_final)

		print('<b>--> Processing DONE !</b>')

		elapsed_time = time() - start_time
		minutes = int(elapsed_time // 60)
		seconds = int(elapsed_time % 60)
		elapsed_time = 'Elapsed Time for <b>{}</b> : {:02d}:{:02d} min.<br>'.format(name, minutes, seconds)
		print(elapsed_time)
		elapsed_time = re.sub(r"<.*?>", "", elapsed_time)   # Remove HTML tags

		if self.SDR_Testing:
			print("----------------------------------------")
			App.compare.SDR(self.song_output_path, self.output_format, self.Gdrive, self.Best_Compensations, elapsed_time)
		
		if self.BATCH_MODE and not self.DEBUG and not self.PREVIEWS:
			self.CONSOLE.clear_output()
		

		# DEBUG : Examples (with old version of KaraFan 1.0)
		
		# instrum = instrum / self.model_instrum['Compensation']
		# self.Save_Audio("Sub - 1", normalized - (instrum * 1.0235))
		# self.Save_Audio("Sub - 2", normalized - (instrum * 1.0240))
		# self.Save_Audio("Sub - 3", normalized - (instrum * 1.0245))


	def Extract_with_Model(self, type, audio, model):
		"""
		Explication from "Jarredou" about the 2 passes :

		This helps reduce/remove the noise added by the MDX models,
		since the phase is inverted before processing and restored afterward in one of the two passes.
		When they are added together, only the MDX noise is out of phase and gets removed,
		while the rest regains its original gain (0.5 + 0.5).
		ZFTurbo also added this to Demucs in the original MVSep-MDX23 code.
		"""
		name = model['Name']

		if type == 'Vocal':		bigshifts = self.shifts_vocals;  text = f'► Extract Vocals with "{name}"'
		elif type == 'Music':	bigshifts = self.shifts_instru;  text = f'► Extract Music with "{name}"'
		elif type == 'Filter':	bigshifts = self.shifts_filter;  text = f'► Clean Vocals with "{name}"'
		
		if not self.large_gpu:
			# print(f'Large GPU is disabled : Loading model "{name}" now...')
			self.Load_MDX(model)
		
		mdx_model = self.MDX[name]['model']
		inference = self.MDX[name]['inference']

		# ONLY 1 Pass, for testing purposes
		if self.TEST_MODE:
			print(text + " (<b>1 Pass !</b>)")
			source = self.demix_full(audio, mdx_model, inference, bigshifts)[0]
		else:
			print(text)
			source = 0.5 * -self.demix_full(-audio, mdx_model, inference, bigshifts)[0]

			source += 0.5 * self.demix_full( audio, mdx_model, inference, bigshifts)[0]

		# **********************************
		# I don't use it for instrumental, because you win only 0.004 points of SDR !!

		JARREDOU_wanna_play_with_SRS = False
		
		# **********************************

		Process_SRS = (type == 'Vocal' or (type == 'Filter' and model['Stem'] == 'Vocals'))

		# Automatic SRS
		if (Process_SRS or JARREDOU_wanna_play_with_SRS) and model['Cut_OFF'] > 0 and model['Name'] != "Vocal Main":  # Exception !!

			# This is mandatory, I don't know why, but without this,
			# the sample rate DOWN doesn't fit the MDX model Band
			# and produce noise in High Frequencies !! (??)
			# @ 510 Hz -> there is less noise, but badder SDR ?!?!
			# TODO : Test with 14600 Hz models cut-off
			# 
			delta = 810 if type == 'Vocal' else 1220 # Hz

			audio_SRS = App.audio_utils.Change_sample_rate(audio, 'DOWN', self.original_cutoff, model['Cut_OFF'] + delta)
			
			# Limit audio to the same frequency cut-off than MDX model : To avoid SRS noise !! (That helps a little bit)
			audio_SRS = App.audio_utils.Pass_filter('lowpass', model['Cut_OFF'], audio_SRS, self.sample_rate, order = 100)

			# DEBUG
			# self.Save_Audio(type + " - SRS REAL - High", audio_SRS)

			# ONLY 1 Pass, for testing purposes
			if self.TEST_MODE:
				print(text + " -> SRS High (<b>1 Pass !</b>)")
				source_SRS = App.audio_utils.Change_sample_rate(
					self.demix_full(audio_SRS, mdx_model, inference, self.shifts_SRS)[0], 'UP', self.original_cutoff, model['Cut_OFF'] + delta)
			else:
				print(text +" -> SRS High")
				source_SRS = 0.5 * App.audio_utils.Change_sample_rate(
					-self.demix_full(-audio_SRS, mdx_model, inference, self.shifts_SRS)[0], 'UP', self.original_cutoff, model['Cut_OFF'] + delta)

				source_SRS += 0.5 * App.audio_utils.Change_sample_rate(
					self.demix_full( audio_SRS, mdx_model, inference, self.shifts_SRS)[0], 'UP', self.original_cutoff, model['Cut_OFF'] + delta)

			# Check if source_SRS is same size than source
			source_SRS = librosa.util.fix_length(source_SRS, size = source.shape[-1])

			# old formula :  vocals = Linkwitz_Riley_filter(vocals.T, 12000, 'lowpass') + Linkwitz_Riley_filter((3 * vocals_SRS.T) / 4, 12000, 'highpass')
			# *3/4 = Dynamic SRS personal taste of "Jarredou", to avoid too much SRS noise
			# He also told me that 12 Khz cut-off was setted for MDX23C model, but now I use the REAL cut-off of MDX models !

			if type == 'Music':
				# OLD formula from Jarredou
				# Avec cutoff = 17.4khz & -60dB d'atténuation et ordre = 12 --> target freq = 16000 hz (-1640)
				
				cut_freq = 7500 # Hz
				if model['Name'] == "Kim Instrum":  cut_freq = 12000 # Hz

				source = App.audio_utils.Linkwitz_Riley_filter('lowpass',  cut_freq, source,     self.sample_rate, order=4) + \
				  		 App.audio_utils.Linkwitz_Riley_filter('highpass', cut_freq, source_SRS, self.sample_rate, order=4)

				# # new multiband ensemble
				# vocals_low = lr_filter((weights[0] * vocals_mdxb1.T + weights[1] * vocals3.T + weights[2] * vocals_mdxb2.T) / weights.sum(), 12000, 'lowpass', order=12)
				# vocals_mid = lr_filter(lr_filter((2 * vocals_mdxb2.T + 2 * vocals_SRS.T + vocals_demucs.T) / 5, 16500, 'lowpass', order=24), 12000, 'highpass', order=12)
				# vocals_high = lr_filter((vocals_demucs.T + vocals_SRS.T) / 2, 16500, 'highpass', order=24)
				# vocals = (vocals_low + vocals_mid + vocals_high) * 1.0074
			else:			
				source = App.audio_utils.Make_Ensemble('Max', [source, source_SRS])

		# Low SRS for Vocal models
		if Process_SRS and JARREDOU_wanna_play_with_SRS == False:

			cut_freq = 18550 # Hz

			audio_SRS = App.audio_utils.Change_sample_rate(audio, 'UP', self.original_cutoff, cut_freq)
			
			# Limit audio to frequency cut-off (That helps a little bit)
			audio_SRS = App.audio_utils.Pass_filter('lowpass', model['Cut_OFF'], audio_SRS, self.sample_rate, order = 100)

			# DEBUG
			# self.Save_Audio(type + " - SRS REAL - Low", audio_SRS)

			# ONLY 1 Pass, for testing purposes
			if self.TEST_MODE:
				print(text + " -> SRS Low (<b>1 Pass !</b>)")
				source_SRS = App.audio_utils.Change_sample_rate(
					self.demix_full(audio_SRS, mdx_model, inference, self.shifts_SRS)[0], 'DOWN', self.original_cutoff, cut_freq)
			else:
				print(text +" -> SRS Low")
				source_SRS = 0.5 * App.audio_utils.Change_sample_rate(
					-self.demix_full(-audio_SRS, mdx_model, inference, self.shifts_SRS)[0], 'DOWN', self.original_cutoff, cut_freq)

				source_SRS += 0.5 * App.audio_utils.Change_sample_rate(
					self.demix_full( audio_SRS, mdx_model, inference, self.shifts_SRS)[0], 'DOWN', self.original_cutoff, cut_freq)

			# Check if source_SRS is same size than source
			source_SRS = librosa.util.fix_length(source_SRS, size = source.shape[-1])

			source = App.audio_utils.Make_Ensemble('Max', [source, source_SRS])

		# DEBUG : Test different values for SDR Volume Compensation
		if self.DEBUG and self.SDR_Testing:
			Best_Volume = App.compare.SDR_Volumes(type, source, model['Compensation'], self.song_output_path, self.Gdrive)

			if model['Compensation'] != Best_Volume:  model['Compensation'] = Best_Volume

		source = source * model['Compensation']  # Volume Compensation

		# TODO
		# source = App.audio_utils.Remove_High_freq_Noise(source, model['Cut_OFF'])

		if not self.large_gpu:  self.Kill_MDX(name)

		return source
	

	def Load_MDX(self, model):
		name = model['Name']
		if name not in self.MDX:
			self.MDX[name] = {}
			self.MDX[name]['model'] = get_models(self.device, model, model['Stem'])
			self.MDX[name]['inference'] = ort.InferenceSession(
				model['PATH'],
				providers = self.providers,
				provider_options = [{"device_id": 0}]
			)
	
	def Kill_MDX(self, model_name):
		if model_name in self.MDX:
			del self.MDX[model_name]['inference']
			del self.MDX[model_name]['model']
			del self.MDX[model_name]
			gc.collect()

	def raise_aicrowd_error(self, msg):
		# Will be used by the evaluator to provide logs, DO NOT CHANGE
		raise NameError(msg)
	
		
	def Check_Already_Processed(self, key, model_name = ""):
		"""
		if GOD MODE :
			- Check if audio file is already processed, and if so, load it.
			- Return AUDIO loaded, or NONE if not found.
		Else :
			- Return NONE.
		Key :
			index of AudioFiles list or "str" (direct filename for test mode)
		"""
		if not self.GOD_MODE or key not in self.AudioFiles_Debug:  return None

		filename = self.AudioFiles[key]
		if self.DEBUG:  filename = f"{key} - {filename}"
		if model_name != "":  filename += " - ("+ model_name +")"

		match self.output_format:
			case 'PCM_16':	filename += '.wav'
			case 'FLOAT':	filename += '.wav'
			case "FLAC":	filename += '.flac'
			case 'MP3':		filename += '.mp3'

		file = os.path.join(self.song_output_path, filename)
		
		if os.path.isfile(file):
			
			print(filename + " --> Loading ...")
			audio, _ = librosa.load(file, mono=False, sr=self.sample_rate)
			
			# Preview Audio file
			if self.PREVIEWS and self.CONSOLE:  self.Show_Preview(filename, audio)

			return audio
		
		return None
	
	def Save_Audio(self, key, audio, model_name = ""):
		"""
		Key : index of AudioFiles list or "str" (direct filename for test mode)
		if Key is a string, it will force saving !
		"""
		
		# Save only mandatory files if not in DEBUG mode
		if type(key) is int:
			if self.DEBUG:
				if key not in self.AudioFiles_Debug and key not in self.AudioFiles_Mandatory:  return
			else:
				if key not in self.AudioFiles_Mandatory:  return

		if type(key) is int:
			filename = self.AudioFiles[key]
			if self.DEBUG:  filename = f"{key} - {filename}"
		else:
			filename = key

		if model_name != "":  filename += " - ("+ model_name +")"

		match self.output_format:
			case 'PCM_16':	filename += '.wav'
			case 'FLOAT':	filename += '.wav'
			case "FLAC":	filename += '.flac'
			case 'MP3':		filename += '.mp3'

		file = os.path.join(self.song_output_path, filename)
		
		# Save as WAV
		match self.output_format:
			case 'PCM_16':
				sf.write(file, audio.T, self.sample_rate, subtype='PCM_16')
			case 'FLOAT':
				sf.write(file, audio.T, self.sample_rate, subtype='FLOAT')
			case "FLAC":
				sf.write(file, audio.T, self.sample_rate, format='flac', subtype='PCM_24')
			case 'MP3':
				# Convert audio to PCM_16 audio data (bytes)
				audio_tmp = (audio.T * 32768).astype(np.int16)  # 2 ^15

				audio_segment = AudioSegment(
					audio_tmp.tobytes(),
					channels = 2,
					frame_rate = self.sample_rate,
					sample_width = 2  # sample width (in bytes)
				)

				# about VBR/CBR/ABR		: https://trac.ffmpeg.org/wiki/Encode/MP3
				# about ffmpeg wrapper	: http://ffmpeg.org/ffmpeg-codecs.html#libmp3lame-1
				# recommended settings	: https://wiki.hydrogenaud.io/index.php?title=LAME#Recommended_encoder_settings

				# 320k is mandatory, else there is a weird cutoff @ 16khz with VBR parameters = ['-q','0'] !!
				# (equivalent to lame "-V0" - 220-260 kbps , 245 kbps average)
				# And also, parameters = ['-joint_stereo', '0'] (Separated stereo channels)
				# is WORSE than "Joint Stereo" for High Frequencies !
				# So let's use it by default for MP3 encoding !!

				audio_segment.export(file, format='mp3', bitrate='320k', codec='libmp3lame')
		
		# Preview Audio file
		if self.PREVIEWS and self.CONSOLE:  self.Show_Preview(filename, audio)

	def Show_Preview(self, name, audio):

		name = os.path.splitext(name)[0]
		
		with self.CONSOLE:
			audio_mp3 = io.BytesIO()
			audio_mp3.name = "Preview.mp3"
			
			# Get the first 60 seconds of the audio
			audio = audio[:, :int(60.3 * self.sample_rate)]

			# Convert audio to PCM_16 audio data (bytes)
			audio_tmp = (audio.T * 32768).astype(np.int16)  # 2 ^15

			audio_segment = AudioSegment(
				audio_tmp.tobytes(),
				channels = 2,
				frame_rate = self.sample_rate,
				sample_width = 2  # sample width (in bytes)
			)

			# audio_segment.export(audio_mp3, format='mp3', bitrate='192k', codec='libmp3lame')
			audio_segment.export(audio_mp3, format='mp3', bitrate='192k', codec='libshine')
			# audio_mp3.seek(0)

			display(HTML(
				'<div class="player"><div>'+ name +'</div><audio controls preload="metadata" src="data:audio/mp3;base64,' \
				+ base64.b64encode(audio_mp3.getvalue()).decode('utf-8') +'"></audio></div>'))

			# audio_mp3.close()

	def demix_full(self, mix, use_model, infer_session, bigshifts):
		
		step = int(self.chunk_size)
		mix_length = mix.shape[1] / 44100

		if bigshifts < 1:  bigshifts = 1  # must not be <= 0 !
		if bigshifts > int(mix_length):  bigshifts = int(mix_length - 1)
		results = []
		shifts  = [x for x in range(bigshifts)]
		
		# Kept in case of Colab policy change for using GUI
		# and we need back to old "stdout" redirection
		#
		# with self.CONSOLE if self.CONSOLE else stdout_redirect_tqdm() as output:
			# dynamic_ncols is mandatory for stdout_redirect_tqdm()
			# for shift in tqdm(shifts, file=output, ncols=40, unit="Big shift", mininterval=1.0, dynamic_ncols=True):

		# with self.CONSOLE if self.CONSOLE else stdout_redirect_tqdm() as output:
		
		self.Progress.reset(len(shifts), unit="Big shift")

		for shift in shifts:
			
			shift_samples = int(shift * 44100)
			# print(f"shift_samples = {shift_samples}")
			
			shifted_mix = np.concatenate((mix[:, -shift_samples:], mix[:, :-shift_samples]), axis=-1)
			# print(f"shifted_mix shape = {shifted_mix.shape}")
			result = np.zeros((1, 2, shifted_mix.shape[-1]), dtype=np.float32)
			divider = np.zeros((1, 2, shifted_mix.shape[-1]), dtype=np.float32)

			total = 0
			for i in range(0, shifted_mix.shape[-1], step):
				total += 1

				start = i
				end = min(i + self.chunk_size, shifted_mix.shape[-1])
				mix_part = shifted_mix[:, start:end]
				# print(f"mix_part shape = {mix_part.shape}")
				sources = demix_base(mix_part, self.device, use_model, infer_session)
				result[..., start:end] += sources
				# print(f"result shape = {result.shape}")
				divider[..., start:end] += 1
			
			result /= divider
			# print(f"result shape = {result.shape}")
			result = np.concatenate((result[..., shift_samples:], result[..., :shift_samples]), axis=-1)
			results.append(result)

			self.Progress.update()
			
		results = np.mean(results, axis=0)
		return results
	
	#----

def Download_Model(model, models_path, CONSOLE = None, PROGRESS = None):
	
	name		= model['Name']
	repo_file	= model['Repo_FileName']
	filename	= re.sub(r"^(UVR-MDX-NET-|UVR_MDXNET_|\d_)*", "", repo_file)
	file_path	= os.path.join(models_path, filename)

	if not os.path.isfile(file_path):
		print(f'Downloading model : "{name}" ...')

		remote_url = 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/' + repo_file
		try:
			response = requests.get(remote_url, stream=True)
			response.raise_for_status()  # Raise an exception in case of HTTP error code
			
			if response.status_code == 200:
				
				total_size = int(response.headers.get('content-length', 0)) // 1048576  # MB
				PROGRESS.reset(total_size, unit="MB")
				
				with open(file_path, 'wb') as file:

					# Kept in case of Colab policy change for using GUI
					# and we need back to old "stdout" redirection
					#
					# with CONSOLE if CONSOLE else stdout_redirect_tqdm() as output:
					#	with tqdm(
					#		file=output, total=total_size,
					#		unit='B', unit_scale=True, unit_divisor=1024,
					#		ncols=40, dynamic_ncols=True, mininterval=1.0
					#	) as bar:

					for data in response.iter_content(chunk_size=1048576):
						# bar.update(len(data))
						PROGRESS.update()
						file.write(data)
			else:
				print(f'Download of model "{name}" FAILED !!')
				Exit_Notebook()
		
		except (requests.exceptions.RequestException, requests.exceptions.ChunkedEncodingError) as e:
			print(f'Error during Downloading "{name}" !!\n\n{e}')
			if os.path.exists(file_path):  os.remove(file_path)
			Exit_Notebook()
	
	return file_path  # Path to this model


# Redirect "Print" to the console widgets (or stdout)
class CustomPrint:
	def __init__(self, console):
		self.CONSOLE = console

	def write(self, text):
		if self.CONSOLE:
			# We are in GUI
			with self.CONSOLE:
				display(HTML('<div class="console">'+ text +'</div>'))
		else:
			# We are in a terminal
			text = re.sub(r"<br>", "\n", text)  # Convert <br> to \n
			text = re.sub(r"&nbsp;", " ", text) # Replace &nbsp; by spaces
			text = re.sub(r"<.*?>", "", text)   # Remove HTML tags
			sys.__stdout__.write(text)

	def flush(self):
		pass

# Kept in case of Colab policy change for using GUI
# and we need back to old "stdout" redirection
#
# Redirect "Print" with tqdm progress bar
# @contextlib.contextmanager
# def stdout_redirect_tqdm():
# 	orig_out_err = sys.stdout, sys.stderr
# 	try:
# 		sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
# 		yield orig_out_err[0]
# 	# Relay exceptions
# 	except Exception as exc:
# 		raise exc
# 	# Always restore sys.stdout/err if necessary
# 	finally:
# 		sys.stdout, sys.stderr = orig_out_err


def Process(params, config):

	global isColab, KILL_on_END

	sys.stdout = CustomPrint(params['CONSOLE'])

	if len(params['input']) == 0:
		print('Error : You have NO file to process in your "input" folder !!');  return
	
	isColab		= params['isColab']
	KILL_on_END	= (config['BONUS']['KILL_on_END'].lower() == "true")

	model = None
	model = MusicSeparationModel(params, config)

	BATCH_MODE = len(params['input']) > 1

	# Process each audio file
	for file in params['input']:
		
		if not os.path.isfile(file):
			print('Error. No such file : {}. Please check path !'.format(file))
			continue
		
		model.SEPARATE(file, BATCH_MODE)
	
	del model; del params; del file

	Exit_Notebook()


def Exit_Notebook():

	# Free & Release GPU memory
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		torch.cuda.ipc_collect()

	gc.collect()
	
	if KILL_on_END:
		# This trick is copyrigthed by "Captain FLAM" (2023) - MIT License
		# That means you can use it, but you have to keep this comment in your code.
		# After deep researches, I found this trick that nobody found before me !!!
		
		# Kill Colab session, especially to save your credits !!
		if isColab:
			from google.colab import runtime
			runtime.unassign()
		else:
			os._exit(0)  # Kill GPU , especially on Laptop !!


if __name__ == '__main__':
	"""
	Example of usage :
		python inference.py
		--input song1.wav song2.mp3 song3.flac
		--output_format MP3
		--overlap_MDX 0.25
		--chunk_size 500000
		--DEBUG
	"""

	print("For now, the command line is not available !")
	print("Please use the GUI instead !")
	os._exit(0)
	
	m = argparse.ArgumentParser()
	m.add_argument('--input', nargs='+', type=str, help='Input audio file or location. You can provide multiple files at once.', required=True)
	m.add_argument('--output', type=str, help='Output folder location for extracted audio files results.')
	m.add_argument('--use_config', action='store_true', help='Use "Config_PC.ini" instead of specifying all options in command line.', default=False)
	m.add_argument('--output_format', type=str, help='Output audio format : "FLAC" (24 bits), "MP3" (CBR 320 kbps), "PCM_16" or "FLOAT" (WAV - PCM 16 bits / FLOAT 32 bits).', default='FLAC')
#	m.add_argument('--preset_genre', type=str, help='Genre of music to automatically select the best A.I models.', default='Pop Rock')
	m.add_argument('--model_instrum', type=str, help='MDX A.I Instrumental model NAME : Replace "spaces" in model\'s name by underscore "_".', default='Instrum HQ 3')
	m.add_argument('--model_vocals',  type=str, help='MDX A.I Vocals model NAME : Replace "spaces" in model\'s name by underscore "_".', default='Kim Vocal 2')
	m.add_argument('--bigshifts_MDX', type=int, help='Managing MDX "BigShifts" trick value.', default=12)
	m.add_argument('--overlap_MDX', type=float, help='Overlap of splited audio for heavy models. Closer to 1.0 - slower.', default=0.0)
#	m.add_argument('--overlap_MDXv3', type=int, help='MDXv3 overlap', default=8)
	m.add_argument('--chunk_size', type=int, help='Chunk size for ONNX models. Set lower to reduce GPU memory consumption OR if you have GPU memory errors !. Default: 500000', default=500000)
	m.add_argument('--use_SRS', action='store_true', help='Use "SRS" vocal 2nd pass : can be useful for high vocals (Soprano by e.g)', default=False)
	m.add_argument('--large_gpu', action='store_true', help='It will store all models on GPU for faster processing of multiple audio files. Requires more GB of free GPU memory.', default=False)
	m.add_argument('--TEST_MODE', action='store_true', help='For testing only : Extract with A.I models with 1 pass instead of 2 passes.\nThe quality will be badder (due to low noise added by MDX models) !', default=False)
	m.add_argument('--DEBUG', action='store_true', help='This option will save all intermediate audio files to compare with the final result.', default=False)
	m.add_argument('--GOD_MODE', action='store_true', help='Give you the GOD\'s POWER : each audio file is reloaded IF it was created before,\nNO NEED to process it again and again !!\nYou\'ll be warned : You have to delete each file that you want to re-process MANUALLY !', default=False)
	
	params = m.parse_args().__dict__

	# We are on PC
	Project = os.getcwd()  # Get the current path
	Gdrive  = os.path.dirname(Project)  # Get parent directory

	cmd_input = params['input']
	params = {
		'input': cmd_input,
		'Gdrive': Gdrive,
		'Project': Project,
		'isColab': False,
		'CONSOLE': None,
		'Progress': None,
		'PREVIEWS': False,
		'DEV_MODE': DEV_MODE,
	}

	config = App.settings.Load(Gdrive, False)

	
	if params['output'] is None:
		print("Error !! You must specify an output folder !")
		os._exit(0)

	# Create missing folders
	folder = os.path.join(Gdrive, "KaraFan_user")
	os.makedirs(folder, exist_ok=True)
	os.makedirs(os.path.join(folder, "Models"), exist_ok=True)

	Process(params, config)
