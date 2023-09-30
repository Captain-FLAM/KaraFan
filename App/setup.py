#!python3.10

#   MIT License - Copyright (c) 2023 - Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan

import os, gc, subprocess, requests, shutil

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
	
	# Get local version
	with open(os.path.join(Project, "App", "__init__.py"), "r") as version_file:
		Version = version_file.readline().replace("# Version", "").strip()

	# For pip install
	os.chdir(Project)

	if isColab:
		Check_dependencies(True)

		# Temporary fix for old KF version < 3.1

		# Rename the folder
		if os.path.exists(os.path.join(Gdrive, "KaraFan")):
			os.rename(os.path.join(Gdrive, "KaraFan"), os.path.join(Gdrive, "KaraFan_user"))

		# Delete everything except Config files & Models folder
		folder = os.path.join(Gdrive, "KaraFan_user")
		if os.path.exists(folder):
			for file in os.listdir(folder):
				item = os.path.join(folder, file)
				if os.path.isfile(item):
					if file != "Config_Colab.ini" and file != "Config_PC.ini":
						os.remove(item)
				elif os.path.isdir(item):
					if file == "Models":
						if os.path.exists(os.path.join(item, "_PARAMETERS_.csv")):
							os.remove(os.path.join(item, "_PARAMETERS_.csv"))
					elif file == "Multi_Song":
						continue
					else:
						shutil.rmtree(item, ignore_errors=True)
	
	# Create missing folders
	folder = os.path.join(Gdrive, "KaraFan_user")
	os.makedirs(folder, exist_ok=True)
	os.makedirs(os.path.join(folder, "Models"), exist_ok=True)

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
			warning +='\n... and extract it in your KaraFan folder.\n'
			warning +='Then, you have to "Restart" the notebook to use the new version of "KaraFan" !\n\n'
			
			if DEV_MODE:
				print(warning)
			else:
				if os.path.exists(os.path.join(Project, ".git")):
					try:
						subprocess.run(["git", "-C", Project, "pull"], text=True, capture_output=True, check=True)

						if isColab:
							print('\n\nFOR NOW : you have to go AGAIN in Colab menu, "Runtime > Restart and Run all" to use the new version of "KaraFan" !\n\n')
						else:
							Check_dependencies(False)
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


if __name__ == '__main__':

	# We are on PC
	Project = os.getcwd()  # Get the current path
	Gdrive  = os.path.dirname(Project)  # Get parent directory

	Install({'Gdrive': Gdrive, 'Project': Project, 'isColab': False, 'I_AM_A_DEVELOPER': False})
