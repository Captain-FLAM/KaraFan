
#   MIT License
#
#   Copyright (c) 2023 ZFTurbo - Start the project MVSEP-MDX23 (music separation model)
#   Copyright (c) 2023 Jarredou - Did all the job for Inference !!
#   Copyright (c) 2023 Captain FLAM - Heavily modified ! (GUI, sequential processing, ...)
#
#   https://github.com/ZFTurbo/MVSEP-MDX23-music-separation-model
#   https://github.com/jarredou/MVSEP-MDX23-Colab_v2/
#   https://github.com/Captain-FLAM/KaraFan


import os, gc, io, sys, csv, base64, argparse, requests
import regex as re
import numpy as np
import onnxruntime as ort
import torch, torch.nn as nn

import librosa, soundfile as sf
from pydub import AudioSegment

from time import time
from scipy import signal
from scipy.signal import resample_poly
from ml_collections import ConfigDict

import ipywidgets as widgets
from IPython.display import display, HTML
import contextlib
from tqdm.auto import tqdm  # Auto : Progress Bar in GUI with ipywidgets
from tqdm.contrib import DummyTqdmFile

import App.settings
from App.tfc_tdf_v3 import TFC_TDF_net

class Conv_TDF_net_trim_model(nn.Module):

	def __init__(self, device, target_name, L, model_params, hop=1024):

		super(Conv_TDF_net_trim_model, self).__init__()
		
		self.dim_c = 4
		self.dim_f = model_params['dim_F_set']
		self.dim_t = 2 ** model_params['dim_T_set']
		self.n_fft = model_params['N_FFT_scale']
		self.hop = hop
		self.n_bins = self.n_fft // 2 + 1
		self.chunk_size = hop * (self.dim_t - 1)
		self.window = torch.hann_window(window_length=self.n_fft, periodic=True).to(device)
		self.target_name = target_name

		out_c = self.dim_c * 4 if target_name == '*' else self.dim_c
		self.freq_pad = torch.zeros([1, out_c, self.n_bins - self.dim_f, self.dim_t]).to(device)

		self.n = L // 2

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

	def forward(self, x):
		x = self.first_conv(x)
		x = x.transpose(-1, -2)

		ds_outputs = []
		for i in range(self.n):
			x = self.ds_dense[i](x)
			ds_outputs.append(x)
			x = self.ds[i](x)

		x = self.mid_dense(x)
		for i in range(self.n):
			x = self.us[i](x)
			x *= ds_outputs[-i - 1]
			x = self.us_dense[i](x)

		x = x.transpose(-1, -2)
		x = self.final_conv(x)
		return x

def get_models(device, model_params, primary_stem = 'vocals'):
	# ??? NOT so simple ... ???
	# FFT = 7680  --> Narrow Band
	# FFT = 6144  --> FULL Band
	model = Conv_TDF_net_trim_model(
		device,
		primary_stem,  # I suppose you can use '*' to get both vocals and instrum, with the new MDX23C model ...
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
			sys.exit(1)
	
	return np.array(sources)

def demix_full(mix, device, chunk_size, model, infer_session, overlap=0.2, bigshifts=1, CONSOLE = None):
	step = int(chunk_size * (1 - overlap))
	shift_number = bigshifts # must not be <= 0 !
	if shift_number < 1:
		shift_number = 1

	mix_length = mix.shape[1] / 44100
	if shift_number > int(mix_length):
		shift_number = int(mix_length - 1)
	shifts = [x for x in range(shift_number)]
	results = []
	
	with CONSOLE if CONSOLE else stdout_redirect_tqdm() as output:

		# dynamic_ncols is mandatory for stdout_redirect_tqdm()
		for shift in tqdm(shifts, file=output, ncols=40, unit="Big shift", mininterval=1.0, dynamic_ncols=True):
			
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
				end = min(i + chunk_size, shifted_mix.shape[-1])
				mix_part = shifted_mix[:, start:end]
				# print(f"mix_part shape = {mix_part.shape}")
				sources = demix_base(mix_part, device, model, infer_session)
				result[..., start:end] += sources
				# print(f"result shape = {result.shape}")
				divider[..., start:end] += 1
			
			result /= divider
			# print(f"result shape = {result.shape}")
			result = np.concatenate((result[..., shift_samples:], result[..., :shift_samples]), axis=-1)
			results.append(result)
		
	results = np.mean(results, axis=0)
	return results

class MusicSeparationModel:
	"""
	Doesn't do any separation just passes the input back as output
	"""
	def __init__(self, options):

		# self.Gdrive  = options['Gdrive']
		self.Project = os.path.join(options['Gdrive'], "KaraFan")
		self.CONSOLE = options['CONSOLE']

		self.output			= options['output']
		self.output_format	= options['output_format']
#		self.preset_genre	= options['preset_genre']
		
		self.bigshifts_MDX	= int(options['bigshifts_MDX'])
		self.overlap_MDX	= float(options['overlap_MDX'])
#		self.overlap_MDXv3	= int(options['overlap_MDXv3'])
		self.use_SRS		= options['use_SRS']
		self.large_gpu		= options['large_gpu']

		self.DEBUG		= options['DEBUG']
		self.TEST_MODE	= options['TEST_MODE']
		self.GOD_MODE	= options['GOD_MODE']
		self.PREVIEWS	= options['PREVIEWS']
			
		self.device = 'cpu'
		if torch.cuda.is_available():  self.device = 'cuda:0'
		print("Use device -> " + self.device.upper())
		
		if self.device == 'cpu':
			print("Warning ! CPU is used instead of GPU for processing. Can be very slow !!")
		
		if self.device == 'cpu':
			self.chunk_size = 200000000
			self.providers = ["CPUExecutionProvider"]
		else:
			self.chunk_size = 1000000
			self.providers = ["CUDAExecutionProvider"]

		if 'chunk_size' in options:
			self.chunk_size = int(options['chunk_size'])
		
		if self.overlap_MDX > 0.99:		self.overlap_MDX = 0.99
		if self.overlap_MDX < 0.0:		self.overlap_MDX = 0.0

#		if self.overlap_MDXv3 > 40:		self.overlap_MDXv3 = 40
#		if self.overlap_MDXv3 < 1:		self.overlap_MDXv3 = 1

		if self.bigshifts_MDX > 41:		self.bigshifts_MDX = 41
		if self.bigshifts_MDX < 1:		self.bigshifts_MDX = 1

		# MDX-B models initialization

		self.model_instrum = None
		self.model_vocals  = None
		instrum = options['model_instrum'].replace("_", " ")  # _ is used for command line
		vocals  = options['model_vocals'].replace("_", " ")

		with open(os.path.join(self.Project, "Models", "_PARAMETERS_.csv")) as csvfile:
			reader = csv.DictReader(csvfile)
			for row in reader:
				# ignore "Other" stems for now !
				name = row['Name']
				if name == instrum:   self.model_instrum = row
				elif name == vocals:  self.model_vocals = row
		
		if self.model_instrum is None:
			print("Parameters for this Instrumentals model not found in the CSV !")
			sys.exit(1)
		if self.model_vocals is None:
			print("Parameters for this Vocals model not found in the CSV !")
			sys.exit(1)
		
		# IMPORTANT : Volume Compensations specific for each model AND each song (different re-mastering(s) in Studio) !!!

		self.model_instrum['Compensation'] = float(self.model_instrum['Compensation'])
		self.model_instrum['Band_Cut_OFF'] = int(self.model_instrum['Band_Cut_OFF'])  # TODO : Use it for SRS
		self.model_instrum['N_FFT_scale']  = int(self.model_instrum['N_FFT_scale'])
		self.model_instrum['dim_F_set']    = int(self.model_instrum['dim_F_set'])
		self.model_instrum['dim_T_set']    = int(self.model_instrum['dim_T_set'])

		self.model_vocals['Compensation']  = float(self.model_vocals['Compensation'])
		self.model_vocals['Band_Cut_OFF']  = int(self.model_vocals['Band_Cut_OFF'])  # TODO : Use it for SRS
		self.model_vocals['N_FFT_scale']   = int(self.model_vocals['N_FFT_scale'])
		self.model_vocals['dim_F_set']     = int(self.model_vocals['dim_F_set'])
		self.model_vocals['dim_T_set']     = int(self.model_vocals['dim_T_set'])

		# Download Models
		remote_url	= 'https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/'
		remove		= re.compile(r"^(UVR-MDX-NET-|UVR_MDXNET_|\d_)*")
		
		filename = re.sub(remove, "", self.model_instrum['Repo_FileName'])
		self.model_path_onnx1 = os.path.join(self.Project, "Models", filename)
		if not os.path.isfile(self.model_path_onnx1):
			if not Download_Model(self.model_instrum['Name'], remote_url + self.model_instrum['Repo_FileName'], self.model_path_onnx1, self.CONSOLE):
				print("Download of model for Instrumentals FAILED !!")
				sys.exit(1)

		filename = re.sub(remove, "", self.model_vocals['Repo_FileName'])
		self.model_path_onnx2 = os.path.join(self.Project, "Models", filename)
		if not os.path.isfile(self.model_path_onnx2):
			if not Download_Model(self.model_vocals['Name'], remote_url + self.model_vocals['Repo_FileName'], self.model_path_onnx2, self.CONSOLE):
				print("Download of model for Vocals FAILED !!")
				sys.exit(1)
		
		# Load Models
		if self.large_gpu:
			print("Large GPU mode is enabled : Loading models now...")

			self.mdx_model1 = get_models(self.device, self.model_instrum, primary_stem = 'instrum')
			self.mdx_model2 = get_models(self.device, self.model_vocals,  primary_stem = 'vocals')

			self.infer_session1 = ort.InferenceSession(
				self.model_path_onnx1,
				providers = self.providers,
				provider_options = [{"device_id": 0}],
			)
			self.infer_session2 = ort.InferenceSession(
				self.model_path_onnx2,
				providers = self.providers,
				provider_options = [{"device_id": 0}],
			)

	def raise_aicrowd_error(self, msg):
		# Will be used by the evaluator to provide logs, DO NOT CHANGE
		raise NameError(msg)
	

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

	def Check_Already_Processed(self, file_key, just_check = False):
		"""
		if GOD MODE :
			- Check if audio file is already processed, and if so, load it.
			- Return AUDIO loaded, or NONE if not found.
		Else :
			- Return NONE.
		"""
		filename = self.AudioFiles[file_key]
		if self.DEBUG:
			filename = f"{file_key} - {filename}"

		file = os.path.join(self.song_output_path, filename)
		
		if self.GOD_MODE and os.path.isfile(file):
			
			if just_check:  return True
			
			print(filename + " --> Already processed (loading now...)")
			audio, _ = librosa.load(file, mono=False, sr=self.sample_rate)
			
			# Preview Audio file
			if self.PREVIEWS and self.CONSOLE:  self.Show_Preview(filename, audio)

			return audio
		
		return None
	
	def Save_Audio(self, file_key, audio):
		"""
		file_key : key of AudioFiles list or "str" (direct filename for test mode)
		"""

		if file_key in self.AudioFiles:
			filename = self.AudioFiles[file_key]
			if self.DEBUG:
				filename = f"{file_key} - {filename}"
		else:
			filename = "Unknown"
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

				#
				# about VBR/CBR/ABR		: https://trac.ffmpeg.org/wiki/Encode/MP3
				# about ffmpeg wrapper	: http://ffmpeg.org/ffmpeg-codecs.html#libmp3lame-1
				# recommended settings	: https://wiki.hydrogenaud.io/index.php?title=LAME#Recommended_encoder_settings

				# 320k is mandatory, else there is a weird cutoff @ 16khz with VBR ['-q','0'] !!
				# (equivalent to lame "-V0" - 220-260 kbps , 245 kbps average)
				# and also, ['-joint_stereo', '0'] (Separated stereo channels) is worse than "Joint Stereo" for High Frequencies !
				audio_segment.export(file, format='mp3', bitrate='320k', codec='libmp3lame')
		
		# Preview Audio file
		if self.PREVIEWS and self.CONSOLE:  self.Show_Preview(filename, audio)


	def Extract_with_Model(self, text, audio, stem, bigshifts_divisor = 1, SRS = False):
		"""
		Explication from "Jarredou" about the 2 passes :

		This helps reduce/remove the noise added by the MDX models,
		since the phase is inverted before processing and restored afterward in one of the two passes.
		When they are added together, only the MDX noise is out of phase and gets removed,
		while the rest regains its original gain (0.5 + 0.5).
		ZFTurbo also added this to Demucs in the original MVSep-MDX23 code.

		Jarredou -> I've never really tested whether it's really useful for Demucs or not, though.
		Captain-FLAM -> I've tested it, and it's really useful : suppress noise between ~ -42 dB < -58 dB !
		"""

		if stem == 'vocals':
			model = self.mdx_model2;  infer = self.infer_session2
		else:
			model = self.mdx_model1;  infer = self.infer_session1

		bigshift = self.bigshifts_MDX
		if bigshifts_divisor > 1: bigshift = bigshift // bigshifts_divisor 

		# ONLY 1 Pass, for testing purposes
		if self.TEST_MODE:
			print(text +" (1 Pass)")
			if not SRS:
				source = demix_full(
					audio,
					self.device, self.chunk_size, model, infer, overlap=self.overlap_MDX, bigshifts=bigshift, CONSOLE = self.CONSOLE
				)[0]
			else:
				source = Change_sample_rate( demix_full(
					Change_sample_rate( audio, 5, 4),
					self.device, self.chunk_size, model, infer, overlap=self.overlap_MDX, bigshifts=bigshift, CONSOLE = self.CONSOLE
				)[0], 4, 5)
		else:
			if not SRS:
				print(text +" (Pass 1)")
				source = 0.5 * -demix_full(
					-audio,
					self.device, self.chunk_size, model, infer, overlap=self.overlap_MDX, bigshifts=bigshift, CONSOLE = self.CONSOLE
				)[0]

				print(text +" (Pass 2)")
				source += 0.5 * demix_full(
					audio,
					self.device, self.chunk_size, model, infer, overlap=self.overlap_MDX, bigshifts=bigshift, CONSOLE = self.CONSOLE
				)[0]
			else:
				print(text +" (Pass 1)")
				source = 0.5 * Change_sample_rate( -demix_full(
					Change_sample_rate( -audio, 5, 4),
					self.device, self.chunk_size, model, infer, overlap=self.overlap_MDX, bigshifts=bigshift, CONSOLE = self.CONSOLE
				)[0], 4, 5)

				print(text +" (Pass 2)")
				source += 0.5 * Change_sample_rate( demix_full(
					Change_sample_rate( audio, 5, 4),
					self.device, self.chunk_size, model, infer, overlap=self.overlap_MDX, bigshifts=bigshift, CONSOLE = self.CONSOLE
				)[0], 4, 5)
		
		return source
	

	def Separate_Music_File(self, file):
		"""
		Implements the sound separation for a single sound file
		"""

		name = os.path.splitext(os.path.basename(file))[0]
		if self.CONSOLE:
			print("Go with : <b>" + name + "</b>")
		else:
			print("Go with : " + name)

		# Create a folder based on input audio file's name
		self.song_output_path = os.path.join(self.output, name)
		if not os.path.exists(self.song_output_path): os.makedirs(self.song_output_path)
		
		# TODO : sr = None --> uses the native sampling rate (if 48 Khz or 96 Khz), maybe not good for MDX models ??
		original_audio, self.sample_rate = librosa.load(file, mono=False, sr = 44100)  # Resample to 44.1 Khz
		
		# Convert mono to stereo (if needed)
		if len(original_audio.shape) == 1:
			original_audio = np.stack([original_audio, original_audio], axis=0)

		print(f"Input audio : {original_audio.shape} - Sample rate : {self.sample_rate}")

		format = '.wav'
		if self.output_format in ['FLAC', 'MP3']:  format = '.' + self.output_format.lower()
		
		# In case of changes, don't forget to update the function in GUI !!
		# - on_Del_Vocals_clicked()
		# - on_Del_Music_clicked()
		
		self.AudioFiles = {
			"1"		: "NORMALIZED" + format,
			"2"		: "Music_extract" + format,
			"3"		: "Audio_sub_Music" + format,
			"4_A"	: "Vocals_Narrow_Band" + format,
			"4_B"	: "Vocals_SRS" + format,
			"4_F"	: "Vocals" + format,
			"5_F"	: "Music" + format,
			"6"		: "Bleeding_in_Music" + format
		}
		
		normalized = self.Check_Already_Processed("1")

		if normalized is None:
			print("► Normalizing audio")
			normalized = Normalize(original_audio)

			# Save Normalized audio
			if self.DEBUG:  self.Save_Audio("1", normalized)
		
		# print("► Processing vocals with MDX23C model")

		# sources3 = demix_full_mdx23c(normalized, self.device, self.overlap_MDXv3)
		# vocals3 = (match_array_shapes(sources3['Vocals'], normalized) \
		# 		+ Lowpass_filter(14700, normalized - match_array_shapes(sources3['Instrumental'], normalized), 44100)) / 2
		
		# if self.DEBUG:
		#	self.Save_Audio("Vocals_MDX23C", vocals3)

		instrum = self.Check_Already_Processed("2")
		
		if instrum is None:
			if not self.large_gpu:
				print("(Large GPU mode is disabled : Loading Instrumental model now...)")
				self.mdx_model1 = get_models(self.device, self.model_instrum, primary_stem = 'instrum')
				self.infer_session1 = ort.InferenceSession(
					self.model_path_onnx1,
					providers = self.providers,
					provider_options = [{"device_id": 0}],
				)
			
			text = "► Processing Music"

			instrum = self.Extract_with_Model(text, normalized, 'instrum', bigshifts_divisor = 2)
			
			# Volume Compensation
			instrum = instrum * self.model_instrum['Compensation']
			
			# Apply silence filter
			# instrum = Silent(instrum, self.sample_rate)

			# Save Instrumental extracted
			if self.DEBUG:  self.Save_Audio("2", instrum)

			# Free GPU memory
			if not self.large_gpu:
				del self.infer_session1; del self.mdx_model1; gc.collect()
		

		vocals_substracted = self.Check_Already_Processed("3")
		
		if vocals_substracted is None:
			print("► Substract Music from Original audio")

			vocals_substracted = normalized - instrum

			# Apply silence filter
			vocals_substracted = Silent(vocals_substracted, self.sample_rate)

			if self.DEBUG:  self.Save_Audio("3", vocals_substracted)
		
		# TESTS - Example
		# instrum = instrum / self.model_instrum['Compensation']
		# self.Save_Audio("Sub - 1" + format, normalized - (instrum * 1.0235))
		# self.Save_Audio("Sub - 2" + format, normalized - (instrum * 1.0240))
		# self.Save_Audio("Sub - 3" + format, normalized - (instrum * 1.0245))

		# Load model for processing with both "Vocals_2" and "Vocals_SRS"
		load_model = False
		if not self.large_gpu:
			if self.use_SRS:
				if not self.Check_Already_Processed("4_A", just_check=True) \
				or not self.Check_Already_Processed("4_B", just_check=True) :
					load_model = True
			elif not self.Check_Already_Processed("4_F", just_check=True):
				load_model = True
			if load_model:
				print("(Large GPU mode is disabled : Loading Vocals model now...)")
				self.mdx_model2 = get_models(self.device, self.model_vocals, primary_stem = 'vocals')
				self.infer_session2 = ort.InferenceSession(
					self.model_path_onnx2,
					providers = self.providers,
					provider_options = [{"device_id": 0}],
				)
		
		if self.use_SRS:
			vocals = self.Check_Already_Processed("4_A")  # Vocals without SRS (Narrow Band)
		else:
			vocals = self.Check_Already_Processed("4_F")

		if vocals is None:
			text = "► Processing Vocals"
			if self.use_SRS: text += " without SRS"
			
			vocals = self.Extract_with_Model(text, vocals_substracted, 'vocals')

			# Volume Compensation
			vocals = vocals * self.model_vocals['Compensation']

			if self.DEBUG:
				if self.use_SRS:
					self.Save_Audio("4_A", vocals)
				else:
					self.Save_Audio("4_F", vocals)
		
		if self.use_SRS:
			vocals_SRS = self.Check_Already_Processed("4_B")  # Vocals with SRS

			if vocals_SRS is None:
				text = "► Processing Vocals with Fullband SRS"

				vocals_SRS = self.Extract_with_Model(text, vocals_substracted, 'vocals', bigshifts_divisor = 5, SRS = True)

				# Volume Compensation
				vocals_SRS = vocals_SRS * self.model_vocals['Compensation']

				vocals_SRS = match_array_shapes(vocals_SRS, vocals)

				if self.DEBUG:  self.Save_Audio("4_B", vocals_SRS)

			# *3/4 = Dynamic SRS personal taste of "Jarredou", to avoid too much SRS noise
			# He also told me that 12 Khz cut-off was setted for MDX23C model, but 14 Khz is better for other MDX models
			# old formula :  vocals = Linkwitz_Riley_filter(vocals.T, 12000, 'lowpass') + Linkwitz_Riley_filter((3 * vocals_SRS.T) / 4, 12000, 'highpass')

			vocals = Linkwitz_Riley_filter(vocals.T, 14000, 'lowpass', self.sample_rate) + Linkwitz_Riley_filter(vocals_SRS.T, 14000, 'highpass', self.sample_rate)
			vocals = vocals.T
		
		# Free GPU memory
		if not self.large_gpu and load_model:
			del self.infer_session2; del self.mdx_model2; gc.collect()

		# Save Vocals
		vocals_final = None # IMPORTANT !!

		if self.use_SRS:
			vocals_final = self.Check_Already_Processed("4_F")
		
		if vocals_final is None:
			print("► Save Vocals final !")

			vocals_final = vocals

			# Apply silence filter
			vocals_final = Silent(vocals_final, self.sample_rate)
			
			self.Save_Audio("4_F", vocals_final)

		# Save Music
		instrum_final = self.Check_Already_Processed("5_F")

		if instrum_final is None:
			print("► Save Music final !")

			instrum_final = normalized - vocals_final

			# Apply silence filter : -61 dB !
			instrum_final = Silent(instrum_final, self.sample_rate, threshold_db = -61)

			self.Save_Audio("5_F", instrum_final)

		
		# TESTS - Example
		# vocals_final = vocals_final / self.model_vocals['Compensation']
		# instrum_final_1 = normalized - (vocals_final * 1.0082)
		# instrum_final_2 = normalized - (vocals_final * 1.0085)
		# instrum_final_3 = normalized - (vocals_final * 1.0088)
		# self.Save_Audio("Music - Test 1" + format, instrum_final_1)
		# self.Save_Audio("Music - Test 2" + format, instrum_final_2)
		# self.Save_Audio("Music - Test 3" + format, instrum_final_3)

		# Save Bleeding Vocals/Other in Music
		bleeding = self.Check_Already_Processed("6")

		if bleeding is None:
			print("► Save Bleedings Vocals/Other in Music !")

			# Don't apply silence filter here !!
			bleeding = instrum_final - instrum

			self.Save_Audio("6", bleeding)


		# TESTS - Example
		# instrum = instrum / self.model_instrum['Compensation']  # Volume Compensation
		# bleeding_1 = instrum_final_1 - instrum
		# bleeding_2 = instrum_final_2 - instrum
		# bleeding_3 = instrum_final_3 - instrum
		# self.Save_Audio("Bleedings - Test 1" + format, bleeding_1)
		# self.Save_Audio("Bleedings - Test 2" + format, bleeding_2)
		# self.Save_Audio("Bleedings - Test 3" + format, bleeding_3)


def Download_Model(name, remote_url, local_path, CONSOLE = None):
	
	print(f'Downloading model : "{name}" ...')
	try:
		response = requests.get(remote_url, stream=True)
		response.raise_for_status()  # Raise an exception in case of HTTP error code
		
		if response.status_code == 200:
			total_size = int(response.headers.get('content-length', 0))
			with open(local_path, 'wb') as file:
				with CONSOLE if CONSOLE else stdout_redirect_tqdm() as output:
					with tqdm(
						file=output, total=total_size,
						unit='B', unit_scale=True, unit_divisor=1024,
						ncols=40, dynamic_ncols=True, mininterval=1.0
					) as bar:
						for data in response.iter_content(chunk_size=1024):
							bar.update(len(data))
							file.write(data)
			return True
		else:
			return False
	
	except (requests.exceptions.RequestException, requests.exceptions.ChunkedEncodingError) as e:
		print(f"Error during Downloading !!\n{e}")
		if os.path.exists(local_path):  os.remove(local_path)
		return False

def Normalize(audio):
	"""
	Normalize audio to -1.0 dB peak amplitude
	This is mandatory because every process is based on RMS dB levels.
	(Volumes Compensations & audio Substractions)
	"""
	audio = audio.T
	
	# Suppress DC shift (center on 0.0 vertically)
	audio -= np.mean(audio)

	# Normalize audio peak amplitude to -1.0 dB
	max_peak = np.max(np.abs(audio))
	if max_peak > 0.0:
		max_db = 10 ** (-1.0 / 20)  # Convert -1.0 dB to linear scale
		audio /= max_peak
		audio *= max_db

	return audio.T

def Silent(audio, sample_rate, threshold_db = -50):
	"""
	Make silent the parts of audio where dynamic range (RMS) goes below threshold.
	Don't misundertand : this function is NOT a noise reduction !
	Its behavior is to clean the audio from "silent parts" (below -XX dB) to :
	- avoid the MLM model to work on "silent parts", and save GPU time
	- avoid the MLM model to produce artifacts on "silent parts"
	- clean the final audio files from residues of "silent parts"
	"""

	min_size		= int(1.000 * sample_rate)  # 1000 ms
	window_frame	= int(0.010 * sample_rate)  #   10 ms
	fade_duration	= int(0.250 * sample_rate)  #  250 ms
	fade_out		= np.linspace(1.0, 0.0, fade_duration)
	fade_in			= np.linspace(0.0, 1.0, fade_duration)

	start = 0; end = 0; audio_length = audio.shape[1]

	for i in range(0, audio_length, window_frame):
		
		# TODO : Maybe use S=audio (Spectrogram) instead of y=audio ??
		RMS = np.max(librosa.amplitude_to_db(librosa.feature.rms(y=audio[:, i:(i + window_frame)], frame_length=window_frame, hop_length=window_frame)))
		
		if RMS > threshold_db:
			# Clean the "min_size" samples found
			if end - start > min_size:

				# Fade out
				if start > fade_duration:
					audio[:, start:(start + fade_duration)] *= fade_out
					start += fade_duration

				# Fade in
				if end < audio_length - fade_duration:
					audio[:, (end - fade_duration):end] *= fade_in
					end -= fade_duration
		
				# Clean in between
				audio[:, start:end] = 0.0

			start = i
		else:
			end = i + window_frame

	return audio

# Linkwitz-Riley filter
def Linkwitz_Riley_filter(audio, cutoff, filter_type, sample_rate, order=4):
	audio = audio.T
	nyquist = 0.5 * sample_rate
	normal_cutoff = cutoff / nyquist
	b, a = signal.butter(order//2, normal_cutoff, btype=filter_type, analog=False)
	filtered_audio = signal.filtfilt(b, a, audio)
	return filtered_audio.T

# SRS
def Change_sample_rate(data, up, down):
	data = data.T
	# print(f"SRS input audio shape: {data.shape}")
	new_data = resample_poly(data, up, down)
	# print(f"SRS output audio shape: {new_data.shape}")
	return new_data.T

# Lowpass filter
def Lowpass_filter(cutoff, data, sample_rate):
	b = signal.firwin(1001, cutoff, fs=sample_rate)
	filtered_data = signal.filtfilt(b, [1.0], data)
	return filtered_data

def match_array_shapes(array_1:np.ndarray, array_2:np.ndarray):
	if array_1.shape[1] > array_2.shape[1]:
		array_1 = array_1[:,:array_2.shape[1]] 
	elif array_1.shape[1] < array_2.shape[1]:
		padding = array_2.shape[1] - array_1.shape[1]
		array_1 = np.pad(array_1, ((0,0), (0,padding)), 'constant', constant_values=0)
	return array_1


# Redirect "Print" to the console widgets (or stdout)
class CustomPrint:
	def __init__(self, console):
		self.CONSOLE = console

	def write(self, text):
		with self.CONSOLE:
			display(HTML('<div class="console">'+ text +'</div>'))

	def flush(self):
		pass

# Redirect "Print" with tqdm progress bar
@contextlib.contextmanager
def stdout_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err


def Run(options):

	start_time = time()
	
	if 'CONSOLE' in options and not options['CONSOLE'] is None:
		sys.stdout = CustomPrint(options['CONSOLE'])

	model = None
	model = MusicSeparationModel(options)

	# Process each audio file
	for file in options['input']:
		
		if not os.path.isfile(file):
			print('Error. No such file : {}. Please check path !'.format(file))
			continue
		
		model.Separate_Music_File(file)
	
	# Free, Release & Kill GPU !!!
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		torch.cuda.ipc_collect()
	
	elapsed_time = time() - start_time
	minutes = int(elapsed_time // 60)
	seconds = int(elapsed_time % 60)
	print('-> Processing DONE !')
	print('Elapsed Time : {:02d}:{:02d} min.'.format(minutes, seconds))


if __name__ == '__main__':
	"""
	Example of usage :
		python inference.py
		--input mixture.wav mixture1.wav
		--output_format MP3
		--overlap_MDX 0.25
		--chunk_size 500000
		--DEBUG
	"""

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
	
	options = m.parse_args().__dict__

	# We are on a PC : Get the current path and remove last part (KaraFan)
	Gdrive = os.getcwd().replace("KaraFan","").rstrip(os.path.sep)

	if options['use_config'] == True:
		
		cmd_input = options['input']

		config = App.settings.Load(Gdrive, False)
		options = App.settings.Convert_to_Options(config)

		options['input'] = cmd_input
	
	elif options['output'] is None:
		print("Error !! You must specify an output folder !")
		sys.exit(0)

	options['Gdrive'] = Gdrive
	options['CONSOLE'] = None
	options['PREVIEWS'] = False

	Run(options)
