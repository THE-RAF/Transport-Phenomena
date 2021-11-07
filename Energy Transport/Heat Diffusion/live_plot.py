import matplotlib.pyplot as plt

class LivePlot:
	def __init__(self, window_title='Plot', xlabel='x', ylabel='y', line_color='k'):
		self.line_color = line_color
		self.figure = plt.figure()
		self.ax1 = plt.subplot()

		self.xlabel = xlabel
		self.ylabel = ylabel
		
		self.figure.canvas.set_window_title(window_title)
		
		self.pause = 0.000001

	def plot(self, x, y, xlim=[], ylim=[]):
		if xlim:
			self.ax1.set_xlim(xlim)
		if ylim:
			self.ax1.set_ylim(ylim)

		self.ax1.set_xlabel(self.xlabel)
		self.ax1.set_ylabel(self.ylabel)

		self.ax1.plot(x, y, color=self.line_color)
		plt.pause(self.pause)
		plt.cla()
