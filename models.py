from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from util import regression_performance, plot_hyperparameter
import numpy as np

"""
Steps to evaluate a ML technique:
	1. Get the data!
	2. Visualize the data through plots.
	3. Train the model.
	4. Evaluate the model.
		-- Use cross validation (if validation or test data is not available)
		   to tune hyperparameters and/or test model.
		-- Look out for overfitting and underfitting.
		-- Repeat Step 3 to find the best model.
"""

def run_SGD(X, y):
	"""
	Runs Stochastic Gradient Descent on the regression data.

	Parameters
	--------------------
		X -- tuple of length 3, 
			1. numpy matrix of shape (n_1,d), features for training
			2. numpy matrix of shape (n_2,d), features for validation
			3. numpy matrix of shape (n_3,d), features for test
		y -- tuple of length 3,
			1. numpy matrix of shape (n_1,1), targets for training
			2. numpy matrix of shape (n_2,1), targets for validation
			3. numpy matrix of shape (n_3,1), targets for test
	"""

	print 'Examining Stochastic Gradient Descent for Linear Regression...'
	# TODO

	"""
	We do not have any test data so use cross validation on the training set
	to test your model.
	"""
	print 'Done'

def run_PA(X, y):
	"""
	Runs the Passive-Aggressive algorithm on the regression data.

	Parameters
	--------------------
		X -- tuple of length 3, 
			1. numpy matrix of shape (n_1,d), features for training
			2. numpy matrix of shape (n_2,d), features for validation
			3. numpy matrix of shape (n_3,d), features for test
		y -- tuple of length 3,
			1. numpy matrix of shape (n_1,1), targets for training
			2. numpy matrix of shape (n_2,1), targets for validation
			3. numpy matrix of shape (n_3,1), targets for test
	"""

	print 'Examining Passive-Aggressive for Regression...'
	# TODO

	"""
	We do not have any test data so use cross validation on the training set
	to test your model.
	"""
	print 'Done'
