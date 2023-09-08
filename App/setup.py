
#   MIT License - Copyright (c) 2023 Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan

import os, sys, requests, subprocess

def Install(Gdrive, isColab, Fresh_install):
	
	#****************************************************************************************************

	Repo_url    = "https://github.com/Captain-FLAM/KaraFan"
	Version_url = "https://raw.githubusercontent.com/Captain-FLAM/KaraFan/master/App/__init__.py"

	# Needed for both PC and Colab (missing packages)
	Requirements = ["pip", "install", "pydub", "configparse", "ml_collections", "onnxruntime-gpu"]

	# Needed only for PC (already installed on Colab)
	if not isColab:
		Requirements += ["soundfile", "librosa", "ipywidgets", "numpy", "scipy", "tqdm"]

	# Audio		: soundfile, librosa, pydub
	# GUI		: configparse, ipywidgets
	# Inference	: numpy, scipy, tqdm, ml_collections, onnxruntime-gpu
	# MP3 Tags	: mutagen (for future use)
	
	# Note : on PC, We need to install PyTorch_CUDA, not torch ! (already installed on Colab)

	#****************************************************************************************************

	Version = ""; Git_version = "";  New_Version = False

	if not os.path.exists(Gdrive):
		print("ERROR : Google Drive path is not valid !")
		sys.exit(1)
	
	os.chdir(Gdrive)
	Project = os.path.join(Gdrive, "KaraFan")

	# Get local version
	with open(os.path.join(Project, "App", "__init__.py"), "r") as version_file:
		Version = version_file.readline().replace("# Version", "").strip()

	# Auto-Magic update !
	if not Fresh_install:
		try:
			response = requests.get(Version_url)
			if response.status_code == requests.codes.ok:
				Git_version = response.text.split('\n')[0].replace("# Version", "").strip()
			else:
				print("Unable to check version on GitHub ! Maybe you're behind a firewall ?")
		except ValueError as e:
			print("Error processing version data :", e)
		except requests.exceptions.ConnectionError as e:
			print("Connection error while trying to fetch version :", e)

		if Version and Git_version:
			if Git_version > Version:
				print(f'Updating "KaraFan" project to version {Git_version} ...')
				try:
					subprocess.run(["git", "-C", Project, "pull"], text=True, capture_output=True, check=True)

					Version = Git_version;  New_Version = True
					
					if isColab:
						print('NOW, you have to go in Colab menu, "Runtime > Restart runtime and Run all" to use the new version of "KaraFan" !')
						sys.exit(0)

				except subprocess.CalledProcessError as e:
					if e.returncode == 127:
						print('WARNING : Git is not installed on your system !')
						print('... and there is a new version of "KaraFan" available !')
						print('You have to download it manually from :')
						print(Repo_url)
						print('... and extract it in your Google Drive folder.')
					else:
						print("Error during Update :\n" + e.stderr + "\n" + e.stdout)
			else:
				print('"KaraFan" is up to date.')

	# Dependencies already installed ?
	if isColab or New_Version:
		try:
			import onnxruntime
		except:
			print("Installing dependencies... This will take few minutes...", end='')
			try:
				subprocess.run(Requirements, text=True, capture_output=True, check=True)
				print("\rInstallation done !                                     ") # Clean line
			
			except subprocess.CalledProcessError as e:
				print("Error during Install dependencies :\n" + e.stderr + "\n" + e.stdout)
				sys.exit(1)

	return Version

if __name__ == '__main__':

	# We are on a PC : Get the current path and remove last part (KaraFan)
	Gdrive = os.getcwd().replace("KaraFan","").rstrip(os.path.sep)

	Install(Gdrive, False, False)
