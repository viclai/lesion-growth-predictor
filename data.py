from extracted_data import matrix_loader as ml
from util import concentration_time_curve, two_dimensional_slices
import numpy as np

class Data:
	def __init__(self, X=None, y=None):
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

	def __init__(self, perfusion_param=None):
		Data.__init__(self)
		self.perfusion_param = perfusion_param

	def load(self, perfusion_param, patch_radius, sampled_by_rbf=False, 
			 merge=True):
		"""
		Load memory-mapped file(s) into X array of features and y array of
		labels.

		Parameters
		--------------------
			perfusion_param -- string, perfusion parameter
			patch_radius    -- integer, radius of patch size
			sampled_by_rbf  -- boolean, data is sampled by RBF
			merge           -- boolean, data from training and validation is 
			                   merged
		"""

		m = ml.Matrix_loader(self.path_to_dataset)
		train_X, train_y, test_X, test_y = m.load(
			perfusion_param,
			patch_radius,
			sampled_by_rbf
			)

		if merge:
			self.X = np.concatenate((train_X, test_X))
			self.y = np.concatenate((train_y, test_y))
		else:
			self.X = train_X, test_X
			self.y = train_y, test_y
		self.perfusion_param = perfusion_param

	def plot(self, type, **kwargs):
		if type == 'time-concentration':
			concentration_time_curve(self.X, **kwargs)
		elif type == '2d-slices':
			kwargs['parameter_name'] = self.perfusion_param
			two_dimensional_slices(self.X, self.y, **kwargs)
