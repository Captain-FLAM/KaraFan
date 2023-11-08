
# GPUtil - GPU utilization
#
# A Python module for programmically getting the GPU utilization from NVIDA GPUs using nvidia-smi
#
# Author: Anders Krogh Mortensen (anderskm)
# Date:   16 January 2017
# Web:    https://github.com/anderskm/gputil
#
#   MIT License - Copyright (c) 2017 - anderskm
#   MIT License - Copyright (c) 2023 - Heavily modified by Captain FLAM for KaraFan !!

import os, platform, subprocess
from distutils import spawn

nvidia_smi = None
GPUs = []

class GPU:
	def __init__(self, ID, uuid, load, memoryTotal, memoryUsed, memoryFree, driver, gpu_name, serial, display_mode, display_active, temp_gpu):
		self.id = ID
		self.uuid = uuid
		self.load = load
		self.memoryUtil = (memoryUsed / memoryTotal)
		self.memoryTotal = memoryTotal
		self.memoryUsed = memoryUsed
		self.memoryFree = memoryFree
		self.driver = driver
		self.name = gpu_name
		self.serial = serial
		self.display_mode = display_mode
		self.display_active = display_active
		self.temperature = temp_gpu

def safeFloatCast(strNumber):
	try:
		number = float(strNumber)
	except ValueError:
		number = float('nan')
	return number

def getGPUs():
	
	global nvidia_smi, GPUs

	if platform.system() == "Windows":
		# If the platform is Windows and nvidia-smi 
		# could not be found from the environment path, 
		# try to find it from system drive with default installation path
		nvidia_smi = spawn.find_executable('nvidia-smi')
		if nvidia_smi is None:
			nvidia_smi = "%s\\Program Files\\NVIDIA Corporation\\NVSMI\\nvidia-smi.exe" % os.environ['systemdrive']
	else:
		nvidia_smi = "nvidia-smi"
	
	# Get ID, processing and memory utilization for all GPUs
	try:
		gpu_info = subprocess.check_output([nvidia_smi, "--query-gpu=index,uuid,utilization.gpu,memory.total,memory.used,memory.free,driver_version,name,gpu_serial,display_active,display_mode,temperature.gpu", "--format=csv,noheader,nounits"], shell=True, stderr=subprocess.STDOUT)
	except:
		return []
	
	# Parse output
	lines = gpu_info.decode('utf-8').split(os.linesep)  # Split on line break

	GPUs = []
	for g in range( len(lines)-1 ):
		line = lines[g]
		#print(line)
		vals = line.split(', ')
		#print(vals)
		for i in range(12):
			# print(vals[i])
			if (i == 0):
				deviceIds = int(vals[i])
			elif (i == 1):
				uuid = vals[i]
			elif (i == 2):
				gpuUtil = safeFloatCast(vals[i])/100
			elif (i == 3):
				memTotal = safeFloatCast(vals[i])
			elif (i == 4):
				memUsed = safeFloatCast(vals[i])
			elif (i == 5):
				memFree = safeFloatCast(vals[i])
			elif (i == 6):
				driver = vals[i]
			elif (i == 7):
				gpu_name = vals[i]
			elif (i == 8):
				serial = vals[i]
			elif (i == 9):
				display_active = vals[i]
			elif (i == 10):
				display_mode = vals[i]
			elif (i == 11):
				temp_gpu = safeFloatCast(vals[i])
		
		GPUs.append(GPU(deviceIds, uuid, gpuUtil, memTotal, memUsed, memFree, driver, gpu_name, serial, display_mode, display_active, temp_gpu))

	return GPUs

def getStatus():
	
	global nvidia_smi, GPUs

	# Get ID, processing and memory utilization for all GPUs
	try:
		gpu_info = subprocess.check_output([nvidia_smi, "--query-gpu=index,utilization.gpu,memory.total,memory.used,temperature.gpu", "--format=csv,noheader,nounits"], shell=True, stderr=subprocess.STDOUT)
	except:
		return []
	
	# Parse output
	lines = gpu_info.decode('utf-8').split(os.linesep)  # Split on line break

	for g in range( len(lines)-1 ):
		line = lines[g]
		vals = line.split(', ')
		for i in range(5):
			if (i == 0):
				deviceId = int(vals[i])
			elif (i == 1):
				gpuUtil = safeFloatCast(vals[i])/100
			elif (i == 2):
				memTotal = safeFloatCast(vals[i])
			elif (i == 3):
				memUsed = safeFloatCast(vals[i])
			elif (i == 4):
				temp_gpu = safeFloatCast(vals[i]);
		
		for gpu in GPUs:
			if (gpu.id == deviceId):
				gpu.load = gpuUtil
				gpu.memoryUtil = (memUsed / memTotal)
				gpu.memoryUsed = memUsed
				gpu.temperature = temp_gpu
				break

	return GPUs