#!python3.10

#   MIT License - Copyright (c) 2023 - Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan

# import ipywidgets as widgets
import time

class Bar:
	def __init__(self, progress_bar, progress_text, unit='Pass'):
		self.value = 0
		self.total = 0
		self.unit = unit
		self.start_time = time.time()
		self.units_time = 0

		# Utilisez les widgets de barre de progression et de boîte de texte fournis
		self.progress_bar  = progress_bar
		self.progress_text = progress_text
		self.progress_text.value = "[00:00:00] - &nbsp;&nbsp;0% - 0/0 - 0.00 sec. / " + unit

	def reset(self, total, unit=''):
		self.value = 0
		self.total = total
		self.unit = unit
		self.progress_bar.value  = 0
		self.progress_bar.max = total
		self.units_time = time.time()

		elapsed_time = time.time() - self.start_time
		self.progress_text.value = f"[{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}] - &nbsp;&nbsp;0% - 0/{self.total} - 0.00 sec. / {self.unit}"

	def update(self, increment=1):
		self.value += increment

		self.progress_bar.value = self.value

		# Update the text of the progress bar
		elapsed_time = time.time() - self.start_time
		units_time   = time.time() - self.units_time
		if self.value > 0:
			time_per_unit = units_time / self.value
		else:
			time_per_unit = 0
		
		# for Download models (Bug : downloaded packets : 34/33 = 101 %)
		if self.value > self.total: self.value = self.total

		percent = int(100 * self.value / self.total)
		if percent < 10:	percent = f"&nbsp;&nbsp;{percent}"
		elif percent < 100:	percent = f"&nbsp;{percent}"

		download = " MB" if self.unit == "MB" else ""
		self.progress_text.value = f"[{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}] - {percent}% - {self.value}/{self.total}{download} - {time_per_unit:.2f} sec. / {self.unit}"
