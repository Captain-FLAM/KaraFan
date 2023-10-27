
#   MIT License - Copyright (c) 2023 - Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan

import os, gc, platform, requests, subprocess, tempfile, zipfile

def Check_dependencies(isColab):

	# Dependencies already installed ?
	print("Installing dependencies... This will take few minutes...", end='')
	try:
		subprocess.run(["pip", "install", "-r", "requirements.txt"], text=True, capture_output=True, check=True)
		if not isColab:
			subprocess.run(["pip", "install", "-r", "requirements_PC.txt"], text=True, capture_output=True, check=True)
		
		print("\rInstallation done !                                     ") # Clean line
	
	except subprocess.CalledProcessError as e:
		print("Error during Install dependencies :\n" + e.stderr + "\n" + e.stdout + "\n")
		Exit_Notebook(isColab)

def Install(params):
	
	Repository  = "https://github.com/Captain-FLAM/KaraFan"
	Version_url = "https://raw.githubusercontent.com/Captain-FLAM/KaraFan/master/App/__init__.py"

	Version = ""; Git_version = ""

	Gdrive = params['Gdrive']
	Project = params['Project']
	isColab = params['isColab']
	DEV_MODE = params['I_AM_A_DEVELOPER']

	if not os.path.exists(Gdrive):
		print("ERROR : Google Drive path is not valid !\n")
		Exit_Notebook(isColab)
	
	# Create missing folders
	user_folder = os.path.join(Gdrive, "KaraFan_user")
	os.makedirs(user_folder, exist_ok=True)
	os.makedirs(os.path.join(user_folder, "Models"), exist_ok=True)

	os.chdir(Project)  # For pip install

	if isColab:
		Check_dependencies(True)
	
	# Get FFmpeg from GitHub wiki
	if not isColab:
		ffmpeg = os.path.join(user_folder, "ffmpeg") + (".exe" if platform.system() == 'Windows' else "")

		if not os.path.exists(ffmpeg):
			print("Downloading FFmpeg... This will take few seconds...", end='')
			try:
				response = requests.get(Repository + '/wiki/FFmpeg/' + platform.system() + '.zip')
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
					print("\nUnable to download FFmpeg from GitHub wiki !")
					Exit_Notebook(isColab)
			except ValueError as e:
				print("\nError processing FFmpeg data :", e)
				Exit_Notebook(isColab)
			except requests.exceptions.ConnectionError as e:
				print("\nConnection error while trying to fetch FFmpeg :", e)
				Exit_Notebook(isColab)
			
			print("\rFFmpeg downloaded !                                     ") # Clean line
			
	# Get local version
	with open(os.path.join(Project, "App", "__init__.py"), "r") as version_file:
		Version = version_file.readline().replace("# Version", "").strip()

	# Auto-Magic update !
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
			print(f'A new version of "KaraFan" is available : {Git_version} !')

			warning = 'You have to download the new version manually from :\n'
			warning += Repository
			warning +='\n... and extract it in your Project folder.\n'
			warning +='Then, you have to "Restart" the notebook to use the new version of "KaraFan" !\n\n'
			
			if DEV_MODE:
				print(warning)
			else:
				if os.path.exists(os.path.join(Project, ".git")):
					try:
						subprocess.run(["git", "-C", Project, "pull"], text=True, capture_output=True, check=True)

						Check_dependencies(isColab)

						if isColab:
							print('\n\nFOR NOW : you have to go AGAIN in Colab menu, "Runtime > Restart and Run all" to use the new version of "KaraFan" !\n\n')
						else:
							print('\n\nFOR NOW : you have to "Restart" the notebook to use the new version of "KaraFan" !\n\n')

						Exit_Notebook(isColab)
						
					except subprocess.CalledProcessError as e:
						if e.returncode == 127:
							print('WARNING : "Git" is not installed on your system !\n' + warning)
						else:
							print("Error during Update :\n" + e.stderr + "\n" + e.stdout)
							Exit_Notebook(isColab)
		else:
			print('"KaraFan" is up to date.')

def Exit_Notebook(isColab):
	gc.collect()
	# This trick is copyrigthed by "Captain FLAM" (2023) - MIT License
	# That means you can use it, but you have to keep this comment in your code.
	# After deep researches, I found this trick that nobody found before me !!!
	if isColab:
		from google.colab import runtime
		runtime.unassign()
	else:
		os._exit(0)


if __name__ == '__main__':

	# We are on PC
	Project = os.getcwd()  # Get the current path
	Gdrive  = os.path.dirname(Project)  # Get parent directory

	Install({'Gdrive': Gdrive, 'Project': Project, 'isColab': False, 'I_AM_A_DEVELOPER': False})
