from extracted_data import matrix_loader as ml
from util import two_dimensional_slices, label_distribution, \
				 scatter_matrix, statistics
import numpy as np
import matplotlib.pyplot as plt

class Data:
	def __init__(self, X=None, y=None):
		"""
        Data class.
        
        Attributes
        --------------------
            X -- numpy matrix of shape (n,d), features
            y -- numpy matrix of shape (n,1), targets
        """

		# n = number of examples, d = dimensionality
		self.X = X
		self.y = y

	def load(self):
		pass

	def plot(self):
		pass

class PerfusionData(Data):
	perfusion_params = list(ml.Matrix_loader().mets.keys())
	patch_radiuses = [int(r[0]) for r in ml.Matrix_loader().halfwindowsizes]
	path_to_dataset = 'extracted_data/gaus_7x7'

	def __init__(self, perfusion_param=None, patch_radius=None):
		"""
        PerfusionData class.

        Attributes
        --------------------
            perfusion_param -- string, perfusion parameter
            patch_radius    -- integer, patch radius
        """

		Data.__init__(self)
		self.perfusion_param = perfusion_param
		self.patch_radius = patch_radius

	def load(self, perfusion_param, patch_radius, sampled_by_rbf=False):
		"""
		Load memory-mapped files into X array of features and y array of
		labels.

		Parameters
		--------------------
			perfusion_param -- string, perfusion parameter
			patch_radius    -- integer, radius of patch size
			sampled_by_rbf  -- boolean, data is sampled by RBF
		"""

		m = ml.Matrix_loader(self.path_to_dataset)
		train_X, train_y, test_X, test_y = m.load(
			perfusion_param,
			patch_radius,
			sampled_by_rbf
			)

		# Merge data
		self.X = np.concatenate((train_X, test_X))
		self.y = np.concatenate((train_y, test_y))
		self.perfusion_param = perfusion_param
		self.patch_radius = patch_radius

	def concentration_time_curve(self, X, **kwargs):
		"""
		Plots the CTC for each instance.

		Parameters
		--------------------
			X -- numpy matrix of shape (n,d), features
		"""

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
			plt.title('Concentration Time Curve')
			plt.draw()
			plt.pause(0.001)
			cont = raw_input("Press [C] to see next curve. ")
			if cont != "C" and cont != "c":
				break
			plt.clf()

	def tissue_curve(self, **kwargs):
		"""
		Plots the CTC of the AIF for each instance.
		"""

		# All columns excluding last 40 columns of entire data set
		instances = self.X.shape[1] - 40
		self.concentration_time_curve(self.X[:,:instances], **kwargs)

	def aterial_input_function(self, **kwargs):
		"""
		Plots the CTC of the AIF for each instance.
		"""

		instances = 40 # Last 40 columns of entire data set
		self.concentration_time_curve(self.X[:,-instances:], **kwargs)

	def plot(self, type, **kwargs):
		"""
		Plots a specified graph to visualize the data.

		Parameters
		--------------------
			type     -- string, type of graph/histogram to plot
		"""

		if type == 'ctc':
			# Plot tissue curve with AIF curve
			self.concentration_time_curve(self.X, **kwargs)
		elif type == 'tissue_curve':
			self.tissue_curve(**kwargs)
		elif type == 'aif':
			self.aterial_input_function(**kwargs)
		elif type == '2d-slices':
			kwargs['parameter_name'] = self.perfusion_param
			kwargs['x-label'] = 'Concentration Level'
			two_dimensional_slices(self.X, self.y, **kwargs)
		elif type == 'label_distribution':
			kwargs['parameter_name'] = self.perfusion_param
			label_distribution(self.y, **kwargs)
		elif type == 'scatter_matrix':
			time_interval = 2
			kwargs['features'] = ['Time ' + str(i) for i in 
				xrange(0, time_interval * self.X.shape[1], time_interval)
				]
			scatter_matrix(self.X, **kwargs)
		elif type == 'stats':
			time_interval = 2
			digits = str(len(str(time_interval * self.X.shape[1])))
			zeros_format = '0' + digits + 'd'
			kwargs['features'] = ['Time ' + format(i, zeros_format) for i in
				xrange(0, time_interval * self.X.shape[1], time_interval)
			]
			filename = ('stats-' + self.perfusion_param + '_' + 
						str(self.patch_radius) + '.csv')
			statistics(self.X, self.y, filename=filename, **kwargs)
		else:
			print 'No such plot exists\n'
