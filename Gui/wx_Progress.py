
#   MIT License - Copyright (c) 2023 - Captain FLAM
#
#   https://github.com/Captain-FLAM/KaraFan

import time, wx

class Bar:
	def __init__(self, GUI):
		self.GUI   = GUI
		self.value = 0
		self.total = 0
		self.unit  = "Pass"
		self.units_time = 0
		self.start_time = time.time()

		wx.CallAfter(self.UpdateProgressBar, 0, "[00:00:00] -   0% - 0/0 - 0.00 sec./ " + self.unit)

	def reset(self, total, unit):
		# Prevent Bug when closing the app (this is an event)
		if wx.GetApp() is None:  return

		self.value = 0
		self.total = total
		self.unit = unit
		self.units_time = time.time()

		elapsed_time = time.time() - self.start_time

		wx.CallAfter(self.UpdateProgressBar,
			0,
			f"[{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}] -   0% - 0/{self.total} - 0.00 sec./ {self.unit}",
			self.total  # Set Range
		)

	def update(self, increment = 1):
		# Prevent Bug when closing the app (this is an event)
		if wx.GetApp() is None:  return
		
		self.value += increment

		# Update the text of the progress bar
		elapsed_time = time.time() - self.start_time
		units_time   = time.time() - self.units_time
		if self.value > 0:
			time_per_unit = units_time / self.value
		else:
			time_per_unit = 0
		
		# for Download models (Bug : downloaded packets : 34/33 = 101 %)
		if self.value > self.total: self.value = self.total

		percent  = int(100 * self.value / self.total)
		if percent < 10:	percent = f"  {percent}"
		elif percent < 100:	percent = f" {percent}"
		else:				percent = f"{percent}"

		download = " MB" if self.unit == "MB" else ""
		
		wx.CallAfter(self.UpdateProgressBar,
			self.value,
			f"[{time.strftime('%H:%M:%S', time.gmtime(elapsed_time))}] - {percent}% - {self.value}/{self.total}{download} - {time_per_unit:.2f} sec./ {self.unit}"
		)

	def UpdateProgressBar(self, value, text, Set_Range = None):
		if Set_Range != None:
			self.GUI.Progress_Bar.SetRange(Set_Range)

		self.GUI.Progress_Bar.Value = value
		self.GUI.Progress_Text.SetLabel(text)
