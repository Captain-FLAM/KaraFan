
#   MIT License - Copyright (c) 2023 - Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan

import os, platform, subprocess, tempfile, zipfile

#*************************************************
#****        DEBUG  ->  for DEVELOPERS        ****
#*************************************************

# Change to True if you're running on your PC (not in Visual Studio Code)
# ... and you don't want Auto-Updates !!

I_AM_A_DEVELOPER = False

#*************************************************

Repository  = "https://github.com/Captain-FLAM/KaraFan"
Version_url = "https://raw.githubusercontent.com/Captain-FLAM/KaraFan/master/App/__init__.py"

# We are on PC - Get the current path of this script
Gdrive  = os.getcwd()

Version = ""; Git_version = ""

# Create missing folders
user_folder = os.path.join(Gdrive, "KaraFan_user")
os.makedirs(user_folder, exist_ok=True)
os.makedirs(os.path.join(user_folder, "Models"), exist_ok=True)

# Temporary fix for old KaraFan < 5.1
old_version = os.path.join(Gdrive, "KaraFan-master")
if os.path.exists(old_version):
	for file in os.listdir(old_version):
		os.remove(os.path.join(old_version, file))
	os.removedirs(old_version)
	
# Dependencies needed at first !
try:
	import requests
except ImportError:
	print("Installing minimal dependencies...", end='')
	subprocess.run(["py", "-3.10", "-m", "pip", "install", "requests"], text=True, capture_output=True, check=True)
	print("\rMinimal Installation done !       ") # Clean line
	import requests

except subprocess.CalledProcessError as e:
	print("\nError during Install dependencies :\n" + e.stderr + "\n" + e.stdout + "\n")
	os._exit(0)

Project = os.path.join(Gdrive, "KaraFan")
Install = False

# Get the latest version
try:
	response = requests.get(Version_url)
	if response.status_code == requests.codes.ok:
		Git_version = response.text.split('\n')[0].replace("# Version", "").strip()
	else:
		print("Unable to check version on GitHub ! Maybe you're behind a firewall ?")
		os._exit(0)
except ValueError as e:
	print("Error processing version data :", e)
	os._exit(0)
except requests.exceptions.ConnectionError as e:
	print("Connection error while trying to fetch version :", e)
	os._exit(0)

if not os.path.exists(Project):
	Install = True  # First install !
else:
# Auto-Magic update !
	# Get local version
	with open(os.path.join(Project, "App", "__init__.py"), "r") as version_file:
		Version = version_file.readline().replace("# Version", "").strip()
	if Version and Git_version:
		if Git_version > Version:
			print(f'A new version of "KaraFan" is available : {Git_version} !')
			
			if I_AM_A_DEVELOPER:
				print('\nYou have to download the new version manually from :')
				print( Repository )
				print('... and extract it in your Project folder.\n')
			else:
				Install = True
		else:
			print('"KaraFan" is up to date.')

if Install:
	# Get the latest version from GitHub
	print("Downloading the latest version of KaraFan...")
	try:
		response = requests.get('https://codeload.github.com/Captain-FLAM/KaraFan/zip/refs/tags/v' + Git_version, allow_redirects=True)
		if response.status_code == requests.codes.ok:
			# Create a temporary file
			temp = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
			temp.write(response.content)

			# Remove old files
			if os.path.exists(Project):
				for file in os.listdir(Project):
					os.remove(os.path.join(Project, file))
				os.rmdir(Project)

			# Unzip the temporary file
			with zipfile.ZipFile(temp.name, 'r') as zip_ref:
				zip_ref.extractall(Gdrive)
				zip_ref.close()

			temp.close()
			os.remove(temp.name)

			# Rename the new folder
			os.rename(os.path.join(Gdrive, "KaraFan-" + Git_version), Project)

			# Remove the "Setup" inside
			os.remove(os.path.join(Project, "Setup.py"))

			# Copy the "KaraFan.pyw" to the Parent directory
			os.rename(Project + os.path.sep + "KaraFan.pyw", os.path.dirname(Project) + os.path.sep + "KaraFan.pyw")
		else:
			print("Unable to download the latest version of KaraFan from GitHub !")
			os._exit(0)
	except ValueError as e:
		print("Error processing version data :", e)
		os._exit(0)
	except requests.exceptions.ConnectionError as e:
		print("Connection error while trying to connect :", e)
		os._exit(0)
	
	print("KaraFan installed !")

# Get FFmpeg from GitHub wiki
ffmpeg = os.path.join(user_folder, "ffmpeg") + (".exe" if platform.system() == 'Windows' else "")

if not os.path.exists(ffmpeg):
	Link = Repository + '/wiki/Data/FFmpeg_' + platform.system() + '.zip'
	print("Downloading FFmpeg... -> " + Link)
	try:
		response = requests.get(Link, allow_redirects=True)
		if response.status_code == requests.codes.ok:
			# Create a temporary file
			temp = tempfile.NamedTemporaryFile(suffix='.zip', delete=False)
			temp.write(response.content)

			# Unzip the temporary file
			with zipfile.ZipFile(temp.name, 'r') as zip_ref:
				zip_ref.extractall(user_folder)
				zip_ref.close()

			# Make it executable
			if platform.platform() == 'Linux':
				subprocess.run(["chmod", "777", ffmpeg], text=True, capture_output=True, check=True)

			temp.close()
			os.remove(temp.name)
		else:
			print("Unable to download FFmpeg from GitHub wiki !")
			os._exit(0)
	except ValueError as e:
		print("Error processing FFmpeg data :", e)
		os._exit(0)
	except requests.exceptions.ConnectionError as e:
		print("Connection error while trying to fetch FFmpeg :", e)
		os._exit(0)
	
	print("FFmpeg downloaded !")

os.chdir(Project)  # For pip install

# Dependencies already installed ?
print("Installing dependencies...")
try:
	subprocess.run(["py", "-3.10", "-m", "pip", "install", "-r", "requirements.txt"], text=True, capture_output=True, check=True)
	subprocess.run(["py", "-3.10", "-m", "pip", "install", "-r", "requirements_PC.txt"], text=True, capture_output=True, check=True)
	
	print("Installation done !")

except subprocess.CalledProcessError as e:
	print("Error during Install dependencies :\n" + e.stderr + "\n" + e.stdout + "\n")
	os._exit(0)

Install = False
try:
	import torch
	if not "+cu" in torch.__version__:
		subprocess.run(["py", "-3.10", "-m", "pip", "uninstall", "torch"], text=True, capture_output=True, check=True)
		Install = True
except ImportError:
	Install = True  # Torch not installed

if Install:
	print('Installing "PyTorch CUDA"... It will take a long time... Be patient !')
	try:
		subprocess.run(["py", "-3.10", "-m", "pip", "install", "torch", "--index-url", "https://download.pytorch.org/whl/cu118"], text=True, capture_output=True, check=True)
		
		print("Installation done !\n")
		print('You can now run KaraFan by launching "KaraFan.pyw" !\n')

		# Wait a key to exit
		input("Press Enter to exit...")
		os._exit(0)

	except subprocess.CalledProcessError as e:
		print('Error during Install "PyTorch CUDA" :\n' + e.stderr + "\n" + e.stdout + "\n")
		os._exit(0)
