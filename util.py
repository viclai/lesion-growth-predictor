import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Functions to visualize data, plot graphs, and evaluate models go here.

def concentration_time_curve(X, **kwargs):
	if 'color' not in kwargs :
		kwargs['color'] = 'b'

	curves = X.shape[0]
	points = X.shape[1]
	time_interval = 2 # Number of seconds that separates each concentration 
	                  # level
	time = np.arange(0, points * time_interval, time_interval)
	plt.ion()
	for i in xrange(curves):
		plt.scatter(time, X[i].A1, **kwargs)
		plt.xlabel(r'Time $t$ (s)', fontsize=16)
		plt.ylabel(r'Concentration Level', fontsize=16)
		plt.title('Concentration-Time Curve ' + str(i))
		plt.draw()
		plt.pause(0.001)
		cont = raw_input("Press [C] to see next curve. ")
		if cont != "C" and cont != "c":
			break
		plt.clf()

def scatter_matrix(X, y):
	pass # TODO

def two_dimensional_slices(X, y, **kwargs):
	if 'color' not in kwargs :
		kwargs['color'] = 'b'

	time_instances = X.shape[1]
	y_label = kwargs.pop('parameter_name') + ' Value'
	plt.ion()
	for i in xrange(time_instances):
		plt.scatter(X[:,i].A1, y.A1, **kwargs)
		plt.xlabel(r'Concentration Level', fontsize=16)
		plt.ylabel(y_label, fontsize=16)
		plt.draw()
		plt.pause(0.001)
		cont = raw_input("Press [C] to see next scatter plot. ")
		if cont != "C" and cont != "c":
			break
		plt.clf()
