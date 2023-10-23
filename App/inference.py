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


import os, gc, sys, csv, time, platform, requests, shutil, torch
import regex as re, numpy as np, onnxruntime as ort

import librosa, tempfile

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

		self.normalize			= int(config['AUDIO']['normalize'])
		self.output_format		= config['AUDIO']['output_format']
		self.silent				= int(config['AUDIO']['silent'])
		self.infra_bass			= (config['AUDIO']['infra_bass'].lower() == "true")
		self.chunk_size			= int(config['OPTIONS']['chunk_size'])
		self.DEBUG				= (config['BONUS']['DEBUG'].lower() == "true")
		self.GOD_MODE			= (config['BONUS']['GOD_MODE'].lower() == "true")
		self.large_gpu			= (config['BONUS']['large_gpu'].lower() == "true")

		self.device = 'cpu'
		self.output = os.path.join(self.Gdrive, config['AUDIO']['output'])
		
		if params['isColab']:
			self.ffmpeg = "/bin/ffmpeg"
		else:
			self.ffmpeg = os.path.join(params['Gdrive'], "KaraFan_user", "ffmpeg") + (".exe" if platform.system() == 'Windows' else "")
		
		if torch.cuda.is_available():  self.device = 'cuda:0'
		
		if self.device == 'cpu':
			self.providers = ["CPUExecutionProvider"]
			print('<div style="font-size:18px;font-weight:bold;color:#ff0040;">Warning ! CPU is used instead of GPU for processing.<br>Processing will be very slow !!</div>')
		else:
			self.providers = ["CUDAExecutionProvider"]
			print('<div style="font-size:18px;font-weight:bold;color:#00b32d;">It\'s OK -> GPU is used for processing !!</div>')
		
		# MDX23C 8K
		self.MDX23_overlap = 1
		with open(os.path.join(params['Project'], "Data", "model_2_stem_full_band_8k.yaml")) as file:
			self.MDX23_config = ConfigDict(yaml.load(file, Loader=yaml.FullLoader))

		# Set BigShifts from Speed option
		match config['OPTIONS']['speed']:
			case 'Fastest':
				self.Quality_Vocal = { 'BigShifts': 1, 'BigShifts_SRS': 0 } # 1 + 0 + 0 = 1 pass
				self.Quality_Music = { 'BigShifts': 1, 'BigShifts_SRS': 0 } # + 1  = 2 pass
				self.Quality_Bleed = { 'BigShifts': 1, 'BigShifts_SRS': 0 }
				self.MDX23_overlap = 1
				self.MDX23_bleed = 1
			case 'Fast':
				self.Quality_Vocal = { 'BigShifts': 1, 'BigShifts_SRS': 1 } # 1 + 1 + 1 = 3 pass
				self.Quality_Music = { 'BigShifts': 1, 'BigShifts_SRS': 0 } # + 1  = 4 pass
				self.Quality_Bleed = { 'BigShifts': 1, 'BigShifts_SRS': 1 }
				self.MDX23_overlap = 2
				self.MDX23_bleed = 1
			case 'Medium':
				self.Quality_Vocal = { 'BigShifts': 1, 'BigShifts_SRS': 3 } # 1 + 3 + 1 = 5 pass
				self.Quality_Music = { 'BigShifts': 2, 'BigShifts_SRS': 0 } # + 2  = 7 pass
				self.Quality_Bleed = { 'BigShifts': 2, 'BigShifts_SRS': 0 }
				self.MDX23_overlap = 4
				self.MDX23_bleed = 2
			case 'Slow':
				self.Quality_Vocal = { 'BigShifts': 2, 'BigShifts_SRS': 3 } # 2 + 3 + 1 = 6 pass
				self.Quality_Music = { 'BigShifts': 3, 'BigShifts_SRS': 0 } # + 3  = 9 pass
				self.Quality_Bleed = { 'BigShifts': 2, 'BigShifts_SRS': 1 }
				self.MDX23_overlap = 6
				self.MDX23_bleed = 3
			case 'Slowest':
				self.Quality_Vocal = { 'BigShifts': 2, 'BigShifts_SRS': 4 } # 2 + 4 + 1 = 7 pass
				self.Quality_Music = { 'BigShifts': 4, 'BigShifts_SRS': 0 } # + 4  = 11 pass
				self.Quality_Bleed = { 'BigShifts': 2, 'BigShifts_SRS': 2 }
				self.MDX23_overlap = 8
				self.MDX23_bleed = 4
		
		# MDX-B models initialization

		self.models = { 'music': [], 'vocal': [], 'bleed_music': [] , 'bleed_vocal': [] , 'remove_music': [] }
		self.MDX = {}

		# Load Models parameters
		with open(os.path.join(params['Project'], "Data", "Models.csv")) as csvfile:
			reader = csv.DictReader(csvfile, quoting=csv.QUOTE_ALL)
			for row in reader:
				# ignore "Other" stems for now !
				name = row['Name']
				if name == "" or name == "Name":  continue

				# IMPORTANT : Volume Compensations are specific for each model !!!

				if name == config['PROCESS']['music_1'] or name == config['PROCESS']['music_2']:
					row['Compensation'] = float(row['Vol_Comp'])
					self.models['music'].append(row)
				
				if name == config['PROCESS']['vocal_1'] or name == config['PROCESS']['vocal_2']:
					row['Compensation'] = float(row['Vol_Comp'])
					self.models['vocal'].append(row)

				# Special case for "Bleedings Filter"

				# --> it's a Music model, so look at "self.Quality_Music" for references !!
				if name == config['PROCESS']['bleed_1'] or name == config['PROCESS']['bleed_2']:
					row['Compensation'] = float(row['Vol_Comp'])
					self.models['bleed_music'].append(row)

				# --> it's a Vocal model, so look at "self.Quality_Vocal" for references !!
				if name == config['PROCESS']['bleed_3'] or name == config['PROCESS']['bleed_4']:
					row['Compensation'] = float(row['Vol_Comp'])
					self.models['bleed_vocal'].append(row)

				if name == config['PROCESS']['bleed_5'] or name == config['PROCESS']['bleed_6']:
					row['Compensation'] = float(row['Vol_Comp'])
					self.models['remove_music'].append(row)
					
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
			"Music extract",
			"Vocal extract",
			"Music Bleedings",
			"Vocal Bleedings",
			"Remove Music",
			"Vocal FINAL",
			"Music FINAL",
		]
		self.AudioFiles_Mandatory = [6, 7]  # Vocal FINAL & Music FINAL
		self.AudioFiles_Debug = [1, 2]		# Music & Vocal extract
		
		# DEBUG
		# Reload "Bleedings" files with GOD MODE ... or not !
		self.AudioFiles_Debug.append(3)
		self.AudioFiles_Debug.append(4)
		self.AudioFiles_Debug.append(5)
		
	# ******************************************************************
	# ****    This is the MAGIC RECIPE , the heart of KaraFan !!    ****
	# ******************************************************************

	def SEPARATE(self, audio_file, BATCH_MODE):

		name = os.path.splitext(os.path.basename(audio_file))[0]
		
		#*************************************************
		#****        DEBUG  ->  for DEVELOPERS        ****
		#*************************************************
		
		# Testing SDR :
		# Put some "SDR_song_XXX.flac" from "Gdrive > KaraFan_user > Multi-Song" in your "Music" folder
		# That's all !!
		# (only the song file, not "instrum.flac" or "vocals.flac" from "Stems" folder)

		self.SDR_Testing = name.startswith("SDR_")

		# Set to False for fast testing (1 pass instead of 2 for denoising)
		self.Denoise = True

		# Save intermediate files Example
		# self.Save_Audio("X - Music SUB", normalized - vocal_ensemble)

		#*************************************************

		start_time = time.time()

		self.BATCH_MODE = BATCH_MODE
		self.song_output_path = os.path.join(self.output, name)
		
		# Delete previous files
		if os.path.exists(self.song_output_path):
			if not self.GOD_MODE:
				print("► No GOD MODE : Re-process ALL files ...")
				for file in os.listdir(self.song_output_path):
					if file != "SDR_Results.txt" and file.endswith(self.output_format):
						os.remove(os.path.join(self.song_output_path, file))
		else:
			# Create a folder based on input audio file's name
			os.makedirs(self.song_output_path)
		
		print("Go with : <b>" + name + "</b>")

		original_audio, self.sample_rate = App.audio_utils.Load_Audio(audio_file, 44100)  # Resample to 44.1 Khz
		
		# TODO : Get the cut-off frequency of the input audio
		# self.original_cutoff = App.audio_utils.Find_Cut_OFF(original_audio, self.sample_rate)
		
		self.original_cutoff = self.sample_rate // 2
		
		print(f"{original_audio.shape[1] // 44100} sec. - Rate : {self.sample_rate} Hz / Cut-OFF : {self.original_cutoff} Hz")
		
		# ****  START PROCESSING  ****
		if self.normalize < 0:
			normalized = self.Check_Already_Processed(0)

			if normalized is None:
				print("► Normalizing audio")
				normalized = App.audio_utils.Normalize(original_audio, self.normalize)

				self.Save_Audio(0, normalized)
		else:
			normalized = original_audio
		
		# 1 - Extract Music with MDX models (Pre-Process before Vocals extraction)

		if len(self.models['music']) == 0:
			music_ensemble = None
		else:
			music_extracts = []
			for model in self.models['music']:
				audio = self.Check_Already_Processed(1, model['Name'])
				if audio is None:
					audio = self.Extract_with_Model("Music", normalized, model)
					
					self.Save_Audio(1, audio, model['Name'])
				
				music_extracts.append(audio)
			
			if len(music_extracts) == 1:
				music_ensemble = music_extracts[0]
			else:
				print("► Make Ensemble Music")
				music_ensemble = App.audio_utils.Make_Ensemble('Max', music_extracts)  # Algorithm ?

				del music_extracts;  gc.collect()
		
		# 2 - Extract Vocals with MDX models

		vocal_extracts = []
		vocal_sub = normalized if music_ensemble is None else normalized - music_ensemble

		for model in self.models['vocal']:
			audio = self.Check_Already_Processed(2, model['Name'])
			if audio is None:
				audio = self.Extract_with_Model("Vocal", vocal_sub, model)
				
				self.Save_Audio(2, audio, model['Name'])
			
			vocal_extracts.append(audio)
		
		if len(vocal_extracts) == 1:
			vocal_ensemble = vocal_extracts[0]
		else:
			print("► Make Ensemble Vocals")
			vocal_ensemble = App.audio_utils.Make_Ensemble('Max', vocal_extracts)  # MAX and not average, because it's Vocals !!

			del vocal_extracts;  gc.collect()
		
		# 3 - Pass Vocals through Filters (remove music bleedings)

		if len(self.models['bleed_music']) == 0:
			vocal_final = vocal_ensemble
		else:
			bleed_extracts = []
			
			for model in self.models['bleed_music']:
				audio = self.Check_Already_Processed(3, model['Name'])
				if audio is None:
					audio = self.Extract_with_Model("Bleed_Music", vocal_ensemble, model)

					self.Save_Audio(3, audio, model['Name'])
				
				bleed_extracts.append(audio)
				
			if len(bleed_extracts) == 1:
				bleed_ensemble = bleed_extracts[0]
			else:
				print("► Make Ensemble Music Bleedings")
				
				bleed_ensemble = App.audio_utils.Make_Ensemble('Max', bleed_extracts)
				
			vocal_final = vocal_ensemble - bleed_ensemble

			del vocal_ensemble; del bleed_extracts; del bleed_ensemble; gc.collect()

		# 4 - Get Music by subtracting Vocals from original audio (for instrumental not captured by MDX models)
		
		print("► Get Music by subtracting Vocals from original audio")

		music_sub = normalized - vocal_final

		# 5 - Pass Music SUB through Filters
		
		if len(self.models['bleed_vocal']) == 0:
			music_final = music_sub
		else:
			if self.DEBUG:  self.Save_Audio("4 - Music - SUB", music_sub)

			bleed_extracts = [];  music_extracts = []
			
			# A - Remove Vocal bleedings
			
			for model in self.models['bleed_vocal']:
				audio = self.Check_Already_Processed(4, model['Name'])
				if audio is None:
					audio = self.Extract_with_Model("Bleed_Vocal", music_sub, model)

					self.Save_Audio(4, audio, model['Name'])
				
				bleed_extracts.append(audio)
				
			if len(bleed_extracts) == 1:
				bleed_ensemble = bleed_extracts[0]
			else:
				bleed_ensemble = App.audio_utils.Make_Ensemble('Max', bleed_extracts)
				
			# B - Remove Music bleedings

			if len(self.models['remove_music']) > 0:
				
				for model in self.models['remove_music']:
					audio = self.Check_Already_Processed(5, model['Name'])
					if audio is None:
						audio = self.Extract_with_Model("Bleed_Music", bleed_ensemble, model)

						self.Save_Audio(5, audio, model['Name'])
					
					music_extracts.append(audio)
					
				if len(music_extracts) == 1:
					bleed_ensemble = bleed_ensemble - music_extracts[0]
				else:
					bleed_ensemble = bleed_ensemble - App.audio_utils.Make_Ensemble('Max', music_extracts)  # Algorithm ?

			music_final = music_sub - bleed_ensemble

			del bleed_extracts; del music_extracts; del bleed_ensemble; gc.collect()

		# 7 - FINAL saving
		
		print("► Save Vocals FINAL !")

		# Apply Infra Bass filter
		if self.infra_bass:  vocal_final = App.audio_utils.Pass_filter('highpass', 18, vocal_final, self.sample_rate, order = 100)

		# Apply silence filter
		if self.silent < 0:  vocal_final = App.audio_utils.Silent(vocal_final, self.sample_rate, self.silent)

		self.Save_Audio(6, vocal_final)

		print("► Save Music FINAL !")
		
		# Apply Infra Bass filter
		if self.infra_bass:  music_final = App.audio_utils.Pass_filter('highpass', 18, music_final, self.sample_rate, order = 100)

		# Apply silence filter
		if self.silent < 0:  music_final = App.audio_utils.Silent(music_final, self.sample_rate, self.silent)

		self.Save_Audio(7, music_final)

		print('<b>--> Processing DONE !</b>')

		elapsed_time = time.time() - start_time
		elapsed_time = f"Elapsed Time for <b>{name}</b> : {time.strftime('%H:%M:%S', time.gmtime(elapsed_time))} sec.<br>"
		print(elapsed_time)
		elapsed_time = re.sub(r"<.*?>", "", elapsed_time)   # Remove HTML tags

		if self.SDR_Testing:
			print("----------------------------------------")
			App.compare.SDR(self.song_output_path, self.output_format, self.Gdrive, elapsed_time)
		
		# Clear screen between each song
		if self.BATCH_MODE and not self.DEBUG:  self.CONSOLE.clear_output()
		

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
			case 'Music':			quality = self.Quality_Music;  text = 'Extract Music'
			case 'Vocal':			quality = self.Quality_Vocal;  text = 'Extract Vocals'
			case 'Bleed_Music':		quality = self.Quality_Bleed;  text = 'Remove Music Bleedings in Vocals'
			case 'Bleed_Vocal':		quality = self.Quality_Bleed;  text = 'Remove Vocal Bleedings in Music'
			case 'Remove_Music':	quality = self.Quality_Bleed;  text = 'Remove Music in Vocal Bleedings'
		
		text	  = f'► {text} with "{name}"'

		# ONLY 1 Pass, with MDX23C models !
		if model['Stem'] == "BOTH":
			device = torch.device(self.device)
			mdx23 = App.tfc_tdf.TFC_TDF_net(self.MDX23_config)
			mdx23.load_state_dict(torch.load(model['PATH']))
			mdx23 = mdx23.to(device)
			mdx23.eval()

			overlap = self.MDX23_overlap if (type == "Music" or type == "Vocal") else self.MDX23_bleed

			print(f"{text} (Overlap : {overlap})")
			sources = demix_base_mdxv3(audio, mdx23, device, self.MDX23_config, overlap, self.Progress)
			
			source = sources['Vocals'] if "Vocal" in type else sources['Instrumental']
			
			# Check if source is same size than audio
			source = librosa.util.fix_length(source, size = audio.shape[-1])

			del sources; mdx23 = mdx23.cpu(); del mdx23; gc.collect(); torch.cuda.empty_cache()
		else:
			if not self.large_gpu:
				# print(f'Large GPU is disabled : Loading model "{name}" now...')
				self.Load_MDX(model)
			
			mdx_model = self.MDX[name]['model']
			inference = self.MDX[name]['inference']

			bigshifts = quality['BigShifts'] 
			
			if self.Denoise:
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

					if self.Denoise:
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
					if self.Denoise:
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
			audio, _ = App.audio_utils.Load_Audio(file, self.sample_rate)
			
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

		file = os.path.join(self.song_output_path, filename)
		
		App.audio_utils.Save_Audio(file, audio, self.sample_rate, self.output_format, self.original_cutoff, self.ffmpeg)

	
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
