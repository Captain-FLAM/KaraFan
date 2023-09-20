#!python3.10

#   MIT License - Copyright (c) 2023 Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan

import os, gc, subprocess, requests

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
		Exit_Notebook()

def Install(Gdrive, Project, isColab, DEV_MODE=False):
	
	Repository  = "https://github.com/Captain-FLAM/KaraFan"
	Version_url = "https://raw.githubusercontent.com/Captain-FLAM/KaraFan/master/App/__init__.py"

	Version = ""; Git_version = ""

	if not os.path.exists(Gdrive):
		print("ERROR : Google Drive path is not valid !\n")
		Exit_Notebook()
	
	# Get local version
	with open(os.path.join(Project, "App", "__init__.py"), "r") as version_file:
		Version = version_file.readline().replace("# Version", "").strip()

	# For pip install
	os.chdir(Project)

	if isColab:
		Check_dependencies(True)

		# Temporary fix for old KF version < 1.4
		# Delete everything except Config files & Models folder
		old_version = os.path.join(Gdrive, "KaraFan")
		if os.path.exists(old_version):
			for file in os.listdir(old_version):
				if os.path.isfile(os.path.join(old_version, file)):
					if file != "Config_Colab.ini" and file != "Config_PC.ini":
						os.remove(os.path.join(old_version, file))
				elif os.path.isdir(os.path.join(old_version, file)):
					if file == "Models":
						if os.path.exists(os.path.join(old_version, "Models", "_PARAMETERS_.csv")):
							os.remove(os.path.join(old_version, "Models", "_PARAMETERS_.csv"))
					else:
						os.rmdir(os.path.join(old_version, file))

			# Rename the folder
			os.rename(old_version, os.path.join(Gdrive, "KaraFan_user"))
	
	# Create missing folders
	folder = os.path.join(Gdrive, "KaraFan_user")
	os.makedirs(folder, exist_ok=True)
	os.makedirs(os.path.join(folder, "Models"), exist_ok=True)

	# Auto-Magic update !
	if not DEV_MODE:
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

				if os.path.exists(os.path.join(Project, ".git")):
					try:
						subprocess.run(["git", "-C", Project, "pull"], text=True, capture_output=True, check=True)

						if isColab:
							print('\n\nFOR NOW : you have to go AGAIN in Colab menu, "Runtime > Restart and Run all" to use the new version of "KaraFan" !\n\n')
						else:
							Check_dependencies(False)
							print('\n\nFOR NOW : you have to "Restart" the notebook to use the new version of "KaraFan" !\n\n')

						Exit_Notebook()
						
					except subprocess.CalledProcessError as e:
						if e.returncode == 127:
							print('WARNING : "Git" is not installed on your system !\n' + warning)
						else:
							print("Error during Update :\n" + e.stderr + "\n" + e.stdout)
							Exit_Notebook()
				else:
					print(warning)
			else:
				print('"KaraFan" is up to date.')


def Exit_Notebook():
	gc.collect()
	os._exit(0)


if __name__ == '__main__':

	# We are on PC
	Project = os.getcwd()  # Get the current path
	Gdrive  = os.path.dirname(Project)  # Get parent directory

	Install(Gdrive, Project, False)
