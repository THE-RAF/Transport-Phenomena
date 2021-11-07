from math import *

import matplotlib.pyplot as plt
from matplotlib import style
style.use('seaborn')

from live_plot import LivePlot

live_plot = LivePlot(window_title='Unidimensional Transient Diffusion', xlabel='Width', ylabel='Temperature')


def rod_mean(rod):
	new_rod = []

	for i, value in enumerate(rod):
		if i == 0:
			new_rod.append((value + rod[i+1])/2)
		elif i == len(rod) - 1:
			new_rod.append((rod[i-1] + value)/2)
		else:
			new_rod.append((rod[i-1] + rod[i+1])/2)

	return new_rod


n_points = 100
rod = [sin(2*pi*(i/n_points)) + (1/2)*sin(20*(i/n_points)) for i in range(n_points)]
x_rod = [i for i in range(len(rod))]

for i in range(150):
	live_plot.plot(x_rod, rod, ylim=[-2, 2])
	
	for k in range(1):
		rod = rod_mean(rod)
