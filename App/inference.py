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


import os, gc, sys, csv, time, requests, io, base64, torch
import regex as re, numpy as np, onnxruntime as ort

import librosa, soundfile as sf
from pydub import AudioSegment

# for MDX23C models
import yaml
from ml_collections import ConfigDict

from IPython.display import display, HTML

import App.settings, App.audio_utils, App.compare, App.tfc_tdf

isColab = False
KILL_on_END = False

def get_models(device, model_params, stem):
	# ??? NOT so simple ... ???
	# FFT = 7680  --> Narrow Band
	# FFT = 6144  --> FULL Band
	model = App.tfc_tdf.Conv_TDF_net_trim_model(
		device,
		# I suppose you can use '*' to get both vocals and instrum, with the new MDX23C model ...
		'vocals' if stem == 'Vocals' else 'instrum',
		11,
		model_params
	)
	return [model]

def demix_base_mdxv3(mix, model, device, config, overlap_MDX23, Progress):
		
		mix = torch.tensor(mix, dtype=torch.float32)
		try:
			S = model.num_target_instruments
		except Exception as e:
			S = model.module.num_target_instruments
		
		mdx_window_size = config.inference.dim_t
		
		# batch_size = config.inference.batch_size
		batch_size = 1
		C = config.audio.hop_length * (mdx_window_size - 1)
		H = C // overlap_MDX23
		L = mix.shape[1]
		pad_size = H - (L - C) % H
		mix = torch.cat([torch.zeros(2, C - H, dtype=torch.float32), mix, torch.zeros(2, pad_size + C - H, dtype=torch.float32)], 1)
		mix = mix.to(device)

		chunks = mix.unfold(1, C, H).transpose(0, 1)

		batches = [chunks[i : i + batch_size] for i in range(0, len(chunks), batch_size)]

		Progress.reset(len(batches), unit="Step")

		X = torch.zeros(S, *mix.shape, dtype=torch.float32).to(device) if S > 1 else torch.zeros_like(mix, dtype=torch.float32)

		with torch.cuda.amp.autocast(dtype=torch.float32):  # BUG fix : float16 by default !!
			with torch.no_grad():
				cnt = 0
				for batch in batches:

					x = model(batch)
					x[torch.isnan(x)] = 0.0  # Replace "NaN" by zeros, just by security ...

					for w in x:
						X[..., cnt * H : (cnt * H) + C] += w
						cnt += 1
				
					Progress.update()

		estimated_sources = X[..., C - H:-(pad_size + C - H)] / overlap_MDX23
		
		if S > 1:
			return {k: v for k, v in zip(config.training.instruments, estimated_sources.cpu().numpy())}
		else:
			return estimated_sources.cpu().numpy()

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
				ten = torch.tensor(res, dtype=torch.float32)
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

		self.output_format		= config['AUDIO']['output_format']
		self.normalize			= (config['AUDIO']['normalize'].lower() == "true")
		self.silent				= - int(config['AUDIO']['silent'])
		self.chunk_size			= int(config['OPTIONS']['chunk_size'])
		self.PREVIEWS			= (config['BONUS']['PREVIEWS'].lower() == "true")
		self.DEBUG				= (config['BONUS']['DEBUG'].lower() == "true")
		self.GOD_MODE			= (config['BONUS']['GOD_MODE'].lower() == "true")
		self.large_gpu			= (config['BONUS']['large_gpu'].lower() == "true")

		self.output = os.path.join(self.Gdrive, config['AUDIO']['output'])
		
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

		# MDX23C 8K
		self.MDX23_overlap = 1
		with open(os.path.join(params['Project'], "App", "model_2_stem_full_band_8k.yaml")) as file:
			self.MDX23_config = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))

		# Set BigShifts from Speed option (last +X is for SRS Low : 1 pass, or not for "x0")
		match config['OPTIONS']['speed']:
			case 'Fastest':
				self.Quality_Vocal = { 'CSV': "x0", 'BigShifts': 1, 'BigShifts_SRS': 0 } # 1 + 0 + 0 = 1 pass
				self.Quality_Bleed = { 'CSV': "x0", 'BigShifts': 1, 'BigShifts_SRS': 0 }
				self.Quality_Music = { 'CSV': "x0", 'BigShifts': 1, 'BigShifts_SRS': 0 } # + 1  = 2 pass
				self.MDX23_overlap = 1
			case 'Fast':
				self.Quality_Vocal = { 'CSV': "x1", 'BigShifts': 1, 'BigShifts_SRS': 1 } # 1 + 1 + 1 = 3 pass
				self.Quality_Bleed = { 'CSV': "x1", 'BigShifts': 1, 'BigShifts_SRS': 0 }
				self.Quality_Music = { 'CSV': "x1", 'BigShifts': 1, 'BigShifts_SRS': 0 } # + 1  = 4 pass
				self.MDX23_overlap = 2
			case 'Medium':
				self.Quality_Vocal = { 'CSV': "x2", 'BigShifts': 1, 'BigShifts_SRS': 3 } # 1 + 3 + 1 = 5 pass
				self.Quality_Bleed = { 'CSV': "x2", 'BigShifts': 2, 'BigShifts_SRS': 0 }
				self.Quality_Music = { 'CSV': "x2", 'BigShifts': 2, 'BigShifts_SRS': 0 } # + 2  = 7 pass
				self.MDX23_overlap = 4
			case 'Slow':
				self.Quality_Vocal = { 'CSV': "x3", 'BigShifts': 2, 'BigShifts_SRS': 3 } # 2 + 3 + 1 = 6 pass
				self.Quality_Bleed = { 'CSV': "x3", 'BigShifts': 2, 'BigShifts_SRS': 0 }
				self.Quality_Music = { 'CSV': "x3", 'BigShifts': 3, 'BigShifts_SRS': 0 } # + 3  = 9 pass
				self.MDX23_overlap = 6
			case 'Slowest':
				self.Quality_Vocal = { 'CSV': "x4", 'BigShifts': 2, 'BigShifts_SRS': 4 } # 2 + 4 + 1 = 7 pass
				self.Quality_Bleed = { 'CSV': "x4", 'BigShifts': 2, 'BigShifts_SRS': 0 }
				self.Quality_Music = { 'CSV': "x4", 'BigShifts': 4, 'BigShifts_SRS': 0 } # + 4  = 11 pass
				self.MDX23_overlap = 8
		
		self.Compensation_Vocal_ENS = 1.0
		self.Compensation_Music_SUB = 1.0
		self.Compensation_Music_ENS = 1.0

		# MDX-B models initialization

		self.models = { 'vocal': [], 'music': [], 'bleed': [] }
		self.MDX = {}

		# Load Models parameters
		with open(os.path.join(params['Project'], "App", "Models_DATA.csv")) as csvfile:
			reader = csv.DictReader(csvfile, quoting=csv.QUOTE_ALL)
			for row in reader:
				# ignore "Other" stems for now !
				name = row['Name']
				if name == "" or name == "Name" or row['Use'] == "":  continue

				# IMPORTANT : Volume Compensations are specific for each model !!!
				# Empirical values to get the best SDR !
				# TODO : Need to be checked against each models combinations !!

				# Set Volume Compensation from Quality option for "Ensembles"
				#  ->  MANDATORY to be set in CSV !!
				compensation = float(row['Comp_' + self.Quality_Vocal['CSV']])

				if name == "VOCAL_ENS_x2"   and len(self.models['vocal']) == 2:	self.Compensation_Vocal_ENS = compensation
				elif name == "VOCAL_ENS_x3" and len(self.models['vocal']) > 2:	self.Compensation_Vocal_ENS = compensation
				elif name == "MUSIC_SUB_x1" and len(self.models['vocal']) == 1:	self.Compensation_Music_SUB = compensation  # only 1 VOCAL !
				elif name == "MUSIC_SUB_x2" and len(self.models['vocal']) == 2:	self.Compensation_Music_SUB = compensation  # 2 VOCALS !
				elif name == "MUSIC_SUB_x3" and len(self.models['vocal']) > 2:	self.Compensation_Music_SUB = compensation  # 3 VOCALS !
				elif name == "MUSIC_ENS_x2" and len(self.models['music']) == 2:
					self.Compensation_Music_ENS = float(row['Comp_' + self.Quality_Music['CSV']])
				else:
					if name == config['PROCESS']['vocal_1'] or name == config['PROCESS']['vocal_2'] \
					or name == config['PROCESS']['vocal_3'] or name == config['PROCESS']['vocal_4']:
						row['Compensation'] = compensation
						self.models['vocal'].append(row)

					elif name == config['PROCESS']['music_1'] or name == config['PROCESS']['music_2']:
						row['Compensation'] = float(row['Comp_' + self.Quality_Music['CSV']])
						self.models['music'].append(row)
					
					# Special case for "Bleedings Filter"
					# --> it's a Music model, so look at "self.Quality_Music" for references !!
					if name == config['PROCESS']['bleed_1'] or name == config['PROCESS']['bleed_2']:
						row['Compensation'] = float(row['Comp_x1']) if self.Quality_Bleed['CSV'] in ["x0", "x1"] else float(row['Comp_x3'])
						self.models['bleed'].append(row)

		# Download Models to :
		models_path	= os.path.join(self.Gdrive, "KaraFan_user", "Models")

		for stem in self.models:
			for model in self.models[stem]:				
				model['Cut_OFF']		= int(model['Cut_OFF'])
				model['N_FFT_scale']	= int(model['N_FFT_scale'])
				model['dim_F_set']		= int(model['dim_F_set'])
				model['dim_T_set']		= int(model['dim_T_set'])

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
			"Bleedings",
			"Music extract",
			"Vocal FINAL",
			"Music FINAL",
		]
		self.AudioFiles_Mandatory = [4, 5]  # Vocal FINAL & Music FINAL
		self.AudioFiles_Debug = [1, 3]  # Vocal extract
		
		# DEBUG : Reload "Bleedings" files with GOD MODE ... or not !
		self.AudioFiles_Debug.append(2)
		
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

		self.SDR_Testing = name.startswith("SDR_")

		self.SDR_Compensation_Test = True

		#*************************************************

		start_time = time.time()

		self.BATCH_MODE = BATCH_MODE
		if self.CONSOLE:	print("Go with : <b>" + name + "</b>")
		else:				print("Go with : " + name)

		# Create a folder based on input audio file's name
		self.song_output_path = os.path.join(self.output, name)
		if not os.path.exists(self.song_output_path): os.makedirs(self.song_output_path)
		
		# TODO : sr = None --> uses the native sampling rate (if 48 Khz or 96 Khz), maybe not good for MDX models ??
		original_audio, self.sample_rate = librosa.load(file, mono=False, sr = 44100)  # Resample to 44.1 Khz
		
		# TODO : Get the cut-off frequency of the input audio
		# self.original_cutoff = App.audio_utils.Find_Cut_OFF(original_audio, self.sample_rate)
		
		self.original_cutoff = self.sample_rate // 2
		
		channels = len(original_audio.shape)
		print(f"{'Stereo' if channels == 2 else 'Mono'} - {int(original_audio.shape[1] / 44100)} sec. - Rate : {self.sample_rate} Hz / Cut-OFF : {self.original_cutoff} Hz")
		
		# Convert mono to stereo (if needed)
		if channels == 1:  original_audio = np.stack([original_audio, original_audio], axis=0)
		
		# ****  START PROCESSING  ****

		if self.normalize:
			normalized = self.Check_Already_Processed(0)

			if normalized is None:
				print("► Normalizing audio")
				normalized = App.audio_utils.Normalize(original_audio)

				self.Save_Audio(0, normalized)
		else:
			normalized = original_audio
		
		# 1 - Extract Vocals with MDX models

		vocal_extracts = []
		for model in self.models['vocal']:
			audio = self.Check_Already_Processed(1, model['Name'])
			if audio is None:
				audio = self.Extract_with_Model("Vocal", normalized, model)

				self.Save_Audio(1, audio, model['Name'])
			
			vocal_extracts.append(audio)
		
		if len(vocal_extracts) == 1:
			vocal_ensemble = vocal_extracts[0]
		else:
			print("► Make Ensemble Vocals")
			vocal_ensemble = App.audio_utils.Make_Ensemble('Max', vocal_extracts)  # MAX and not average, because it's Vocals !!

			# DEBUG : Test different values for SDR Volume Compensation
			if self.DEBUG and self.SDR_Testing and self.SDR_Compensation_Test:
				Best_Volume = App.compare.SDR_Volumes("Vocal", vocal_ensemble, self.Compensation_Vocal_ENS, self.song_output_path, self.Gdrive)

				if self.Compensation_Vocal_ENS != Best_Volume:
					self.Compensation_Vocal_ENS = Best_Volume

			vocal_ensemble = vocal_ensemble * self.Compensation_Vocal_ENS

		del vocal_extracts;  gc.collect()
		
		if self.DEBUG and len(self.models['vocal']) > 1 and len(self.models['bleed']) > 0:
			self.Save_Audio("1 - "+ self.AudioFiles[1] +" - Ensemble", vocal_ensemble)

		# 2 - Pass Vocals through Filters (remove music bleedings)

		if len(self.models['bleed']) == 0:
			vocal_final = vocal_ensemble
		else:
			bleed_extracts = []
			
			for model in self.models['bleed']:
				audio = self.Check_Already_Processed(2, model['Name'])
				if audio is None:
					audio = self.Extract_with_Model("Bleed", vocal_ensemble, model)

					# Apply silence filter
					audio = App.audio_utils.Silent(audio, self.sample_rate)

					self.Save_Audio(2, audio, model['Name'])
				
				bleed_extracts.append(audio)
				
			if len(bleed_extracts) == 1:
				bleed_ensemble = bleed_extracts[0]
			else:
				print("► Make Ensemble Bleedings")
				
				bleed_ensemble = App.audio_utils.Make_Ensemble('Max', bleed_extracts)
				
				if self.DEBUG:  self.Save_Audio("2 - "+ self.AudioFiles[2] +" - Ensemble", bleed_ensemble)

			vocal_final = vocal_ensemble - bleed_ensemble

			del vocal_ensemble; del bleed_extracts; del bleed_ensemble; gc.collect()

		# 3 - Get Music by subtracting Vocals from original audio (for instrumental not captured by MDX models)
		
		print("► Get Music by subtracting Vocals from original audio")
		music_sub = normalized - vocal_final

		# DEBUG : Test different values for SDR Volume Compensation
		if self.DEBUG and self.SDR_Testing and self.SDR_Compensation_Test:
			Best_Volume = App.compare.SDR_Volumes("Music", music_sub, self.Compensation_Music_SUB, self.song_output_path, self.Gdrive)

			if self.Compensation_Music_SUB != Best_Volume:
				self.Compensation_Music_SUB = Best_Volume

		music_sub = music_sub * self.Compensation_Music_SUB

		# 4 - Repair Music

		if len(self.models['music']) == 0:
			music_final = music_sub
		else:
			if self.DEBUG:  self.Save_Audio("3 - Music - SUB", music_sub)

			# Extract Music with MDX models
			music_extracts = []
			
			for model in self.models['music']:
				audio = self.Check_Already_Processed(3, model['Name'])
				if audio is None:
					audio = self.Extract_with_Model("Music", normalized, model)

					self.Save_Audio(3, audio, model['Name'])
				
				music_extracts.append(audio)
				
			if len(music_extracts) == 1:
				music_ensemble = music_extracts[0]
			else:
				print("► Make Ensemble Music")
				
				music_ensemble = App.audio_utils.Make_Ensemble('Max', music_extracts)  # Algorithm
				
				# DEBUG : Test different values for SDR Volume Compensation
				if self.DEBUG and self.SDR_Testing and self.SDR_Compensation_Test:
					Best_Volume = App.compare.SDR_Volumes("Music", music_ensemble, self.Compensation_Music_ENS, self.song_output_path, self.Gdrive)

					if self.Compensation_Music_ENS != Best_Volume:
						self.Compensation_Music_ENS = Best_Volume

				music_ensemble = music_ensemble * self.Compensation_Music_ENS

				if self.DEBUG:  self.Save_Audio("3 - "+ self.AudioFiles[3] +" - Ensemble", music_ensemble)

			del music_extracts;  gc.collect()

			print("► Repair Music")

			music_final = App.audio_utils.Make_Ensemble('Max', [music_sub, music_ensemble])  # Algorithm

		# 5 - FINAL saving
		
		print("► Save Vocals FINAL !")

		# Apply silence filter
		vocal_final = App.audio_utils.Silent(vocal_final, self.sample_rate)

		self.Save_Audio(4, vocal_final)

		print("► Save Music FINAL !")
		
		# Apply silence filter
		music_final = App.audio_utils.Silent(music_final, self.sample_rate)

		self.Save_Audio(5, music_final)

		print('<b>--> Processing DONE !</b>')

		elapsed_time = time.time() - start_time
		elapsed_time = f"Elapsed Time for <b>{name}</b> : {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))} sec.<br>"
		print(elapsed_time)
		elapsed_time = re.sub(r"<.*?>", "", elapsed_time)   # Remove HTML tags

		if self.SDR_Testing:
			print("----------------------------------------")
			App.compare.SDR(self.song_output_path, self.output_format, self.Gdrive, elapsed_time)
		
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

		match type:
			case 'Vocal':	quality = self.Quality_Vocal;  text = 'Extract Vocals'
			case 'Music':	quality = self.Quality_Music;  text = 'Extract Music'
			case 'Bleed':	quality = self.Quality_Bleed;  text = 'Clean Bleedings'
		
		text	  = f'► {text} with "{name}"'
		denoise   = (quality['CSV'] != "x0")

		if name.startswith("MDX23C"):
			device = torch.device(self.device)
			mdx23 = App.tfc_tdf.TFC_TDF_net(self.MDX23_config)
			mdx23.load_state_dict(torch.load(model['PATH']))
			mdx23 = mdx23.to(device)
			mdx23.eval()

			# ONLY 1 Pass, with this model !
			print(f"{text} (Overlap : {self.MDX23_overlap})")
			source = demix_base_mdxv3(audio, mdx23, device, self.MDX23_config, self.MDX23_overlap, self.Progress)['Vocals']
			
			# Check if source is same size than audio
			source = librosa.util.fix_length(source, size = audio.shape[-1])

			mdx23 = mdx23.cpu(); del mdx23; gc.collect(); torch.cuda.empty_cache()
		else:
			if not self.large_gpu:
				# print(f'Large GPU is disabled : Loading model "{name}" now...')
				self.Load_MDX(model)
			
			mdx_model = self.MDX[name]['model']
			inference = self.MDX[name]['inference']

			bigshifts = quality['BigShifts']
			
			if denoise:
				print(f"{text} ({quality['BigShifts']} pass)")
				source  = 0.5 * -self.demix_full(-audio, mdx_model, inference, bigshifts)[0]
				source += 0.5 *  self.demix_full( audio, mdx_model, inference, bigshifts)[0]
			else:
				# ONLY 1 Pass, for testing purposes
				print(f"{text} ({quality['BigShifts']} pass) - <b>NO Denoise !</b>")
				source = self.demix_full(audio, mdx_model, inference, bigshifts)[0]

			# Automatic SRS (for not FULL-BAND models !)
			if quality['BigShifts_SRS'] > 0:

				bigshifts = quality['BigShifts_SRS']

				# 1 - High SRS

				if model['Cut_OFF'] > 0 and model['Name'] != "Vocal Main":  # Exception !!

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

					if denoise:
						print(f"{text} -> SRS High ({bigshifts} pass)")
						
						source_SRS = 0.5 * App.audio_utils.Change_sample_rate(
							-self.demix_full(-audio_SRS, mdx_model, inference, bigshifts)[0], 'UP', self.original_cutoff, model['Cut_OFF'] + delta)
						
						source_SRS += 0.5 * App.audio_utils.Change_sample_rate(
							self.demix_full( audio_SRS, mdx_model, inference, bigshifts)[0], 'UP', self.original_cutoff, model['Cut_OFF'] + delta)
					else:
						# ONLY 1 Pass, for testing purposes
						print(f"{text} -> SRS High ({bigshifts} pass) - <b>NO Denoise !</b>")
						source_SRS = App.audio_utils.Change_sample_rate(
							self.demix_full(audio_SRS, mdx_model, inference, bigshifts)[0], 'UP', self.original_cutoff, model['Cut_OFF'] + delta)

					# Check if source_SRS is same size than source
					source_SRS = librosa.util.fix_length(source_SRS, size = source.shape[-1])

					if type == 'Vocal':
						source = App.audio_utils.Make_Ensemble('Max', [source, source_SRS])
					else:
						# OLD formula --> from Jarredou
						
						# vocals = Linkwitz_Riley_filter(vocals.T, 12000, 'lowpass') + Linkwitz_Riley_filter((3 * vocals_SRS.T) / 4, 12000, 'highpass')
						# *3/4 = Dynamic SRS personal taste of "Jarredou", to avoid too much SRS noise
						# He also told me that 12 Khz cut-off was setted for MDX23C model, but now I use the REAL cut-off of MDX models !

						# Avec cutoff = 17640 hz & -60dB d'atténuation et ordre = 12 --> cut freq = 16000 hz (-1640)
						# cut_freq = 14400 # Hz
						# # cut_freq = 7500 # Hz
						# # if model['Name'] == "Kim Instrum":  cut_freq = 12000 # Hz
						source = App.audio_utils.Linkwitz_Riley_filter('lowpass',  16000, source,     self.sample_rate, order=12) + \
								App.audio_utils.Linkwitz_Riley_filter('highpass', 16000, source_SRS, self.sample_rate, order=12)

					# # new multiband ensemble
					# vocals_low = lr_filter((weights[0] * vocals_mdxb1.T + weights[1] * vocals3.T + weights[2] * vocals_mdxb2.T) / weights.sum(), 12000, 'lowpass', order=12)
					# vocals_mid = lr_filter(lr_filter((2 * vocals_mdxb2.T + 2 * vocals_SRS.T + vocals_demucs.T) / 5, 16500, 'lowpass', order=24), 12000, 'highpass', order=12)
					# vocals_high = lr_filter((vocals_demucs.T + vocals_SRS.T) / 2, 16500, 'highpass', order=24)
					# vocals = (vocals_low + vocals_mid + vocals_high) * 1.0074
					
				# 2 - Low SRS -> Bigshifts only 1 pass, else bad SDR
				
				if type == 'Vocal':
					
					cut_freq = 18550 # Hz

					audio_SRS = App.audio_utils.Change_sample_rate(audio, 'UP', self.original_cutoff, cut_freq)
					
					# Limit audio to frequency cut-off (That helps a little bit)
					audio_SRS = App.audio_utils.Pass_filter('lowpass', model['Cut_OFF'], audio_SRS, self.sample_rate, order = 100)

					# DEBUG
					# self.Save_Audio(type + " - SRS REAL - Low", audio_SRS)

					# ONLY 1 Pass, for testing purposes
					if denoise:
						print(f"{text} -> SRS Low (1 pass)")
						
						source_SRS = 0.5 * App.audio_utils.Change_sample_rate(
							-self.demix_full(-audio_SRS, mdx_model, inference, 1)[0], 'DOWN', self.original_cutoff, cut_freq)

						source_SRS += 0.5 * App.audio_utils.Change_sample_rate(
							self.demix_full( audio_SRS, mdx_model, inference, 1)[0], 'DOWN', self.original_cutoff, cut_freq)
					else:
						print(f"{text} -> SRS Low (1 pass) - <b>NO Denoise !</b>")
						source_SRS = App.audio_utils.Change_sample_rate(
							self.demix_full(audio_SRS, mdx_model, inference, 1)[0], 'DOWN', self.original_cutoff, cut_freq)

					# Check if source_SRS is same size than source
					source_SRS = librosa.util.fix_length(source_SRS, size = source.shape[-1])

					source = App.audio_utils.Make_Ensemble('Max', [source, source_SRS])

			if not self.large_gpu:  self.Kill_MDX(name)

		# DEBUG : Test different values for SDR Volume Compensation
		if type != 'Bleed' and self.DEBUG and self.SDR_Testing and self.SDR_Compensation_Test:
			Best_Volume = App.compare.SDR_Volumes(type, source, model['Compensation'], self.song_output_path, self.Gdrive)

			if model['Compensation'] != Best_Volume:  model['Compensation'] = Best_Volume

		source = source * model['Compensation']  # Volume Compensation

		# TODO
		# source = App.audio_utils.Remove_High_freq_Noise(source, model['Cut_OFF'])

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
		
		results = []
		mix_length = int(mix.shape[1] / 44100)

		# Don't depass the length of the song in seconds

		if bigshifts < 1:  bigshifts = 1  # must not be <= 0 !
		if bigshifts > mix_length:  bigshifts = mix_length - 1
		
		# "demix_seconds" is equal to the number of seconds to shift the mix
		# and it's same of "bigshifts" for average results
		demix_seconds = bigshifts
		while bigshifts * demix_seconds > mix_length:  demix_seconds -= 1

		shifts  = [x * demix_seconds for x in range(bigshifts)]
		
		self.Progress.reset(len(shifts), unit="Pass")

		for shift in shifts:
			
			shift_samples = int(shift * 44100)
			# print(f"shift_samples = {shift_samples}")
			
			shifted_mix = np.concatenate((mix[:, -shift_samples:], mix[:, :-shift_samples]), axis=-1)
			# print(f"shifted_mix shape = {shifted_mix.shape}")
			result = np.zeros((1, 2, shifted_mix.shape[-1]), dtype=np.float32)
			divider = np.zeros((1, 2, shifted_mix.shape[-1]), dtype=np.float32)

			total = 0
			for i in range(0, shifted_mix.shape[-1], self.chunk_size):
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

					for data in response.iter_content(chunk_size=1048576):
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
