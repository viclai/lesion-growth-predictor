from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from util import regression_performance, plot_hyperparameter, \
				 plot_incremental_performance, record_results
from sklearn.model_selection import KFold, ShuffleSplit
import numpy as np
import pandas as pd
import os

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
	We do not have any validation data so use cross validation on the training
	set to tune your hyperparameters.
	"""
	print 'Done'

def run_PA(X, y, **kwargs):
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

	total_training_instances = len(X[0])

	if 'parameter' not in kwargs:
		perfusion_param = None
	else:
		perfusion_param = kwargs.pop('parameter')

	if 'patch_radius' not in kwargs:
		patch_radius = None
	else:
		patch_radius = kwargs.pop('patch_radius')

	attributes = [
		'Perfusion Parameter',
		'Model',
		'Patch Radius',
		'Mini Batch Size',
		'Total Number of Examples Trained',
		'Training MSE',
		'Test MSE',
		'Training R^2 Score',
		'Test R^2 Score',
		'C (Regularization)',
		'Epsilon',
		'Fit Intercept?',
		'Number of Epochs',
		'Shuffle?',
		'Random Seed',
		'Loss Function',
		'Warm Start?'
	]
	results = {}
	for a in attributes:
		results[a] = []

	mini_batch_size = 100 # Default
	size = raw_input('Enter mini batch size: ')
	if size == 'q':
		return
	if size != '':
		mini_batch_size = int(size)

	seed = None # Default
	pick_seed = raw_input('Enter seed of random number generator to '
						  'shuffle: ')
	if pick_seed == 'q':
		return
	if pick_seed != '':
		seed = int(pick_seed)

	# Shuffle data
	indices = np.arange(0, total_training_instances)
	np.random.shuffle(indices)
	train_data = np.matrix([np.asarray(X[0])[i] for i in indices])
	outcomes = np.matrix([[y[0].A1[i]] for i in indices])
	
	print 'Tuning regularization C...'
	while True:
		comp = raw_input('Compare errors for range of C values? [Y/n] ')
		#np.random.seed(seed)
		if comp == 'Y':
			start = raw_input('Enter lower bound (inclusive) of range: ')
			end = raw_input('Enter upper bound (exclusive) of range: ')
			incr = raw_input('Enter increment: ')
			C_range = np.arange(float(start), float(end), float(incr))

			avg_train_perf = []
			avg_val_perf = []

			for c in C_range:
				train_perf = []
				val_perf = []

				# Use cross validation to tune parameter
				kf = KFold() # TODO: Use StratifiedKFold
				for train, val in kf.split(train_data):
					X_train, X_val = train_data[train], train_data[val]
					y_train, y_val = outcomes[train].A1, outcomes[val].A1

					model = PassiveAggressiveRegressor(
						C=c,
						random_state=np.random.RandomState(seed)
						)

					for i in xrange(0, len(X_train), mini_batch_size):
						model = model.partial_fit(
							X_train[i:i + mini_batch_size],
							y_train[i:i + mini_batch_size]
							)
					y_pred = model.predict(X_train)
					train_perf.append(regression_performance(
						y_train,
						y_pred,
						'mse'
						))
					y_pred = model.predict(X_val)
					val_perf.append(regression_performance(
						y_val,
						y_pred,
						'mse'
						))
				avg_train_perf.append(
					np.sum(train_perf) * 1.0 / len(train_perf)
					)
				avg_val_perf.append(
					np.sum(val_perf) * 1.0 / len(val_perf)
					)
			plot_hyperparameter(
				C_range,
				avg_train_perf,
				avg_val_perf,
				**{
					'parameter' : r'Regularization $C$',
					'score' : 'Mean Squared Error'
				})
		elif comp == 'q':
			return
		else:
			break

	best_C = 1.0 # Default
	c_pick = raw_input('Choose value of C (default: ' + str(best_C) + '): ')
	if c_pick != '':
		best_C = float(c_pick)


	print 'Tuning epsilon (threshold)...'
	while True:
		comp = raw_input('Compare errors for range of epsilon values? [Y/n] ')
		#np.random.seed(seed)
		if comp == 'Y':
			start = raw_input('Enter lower bound (inclusive) of range: ')
			end = raw_input('Enter upper bound (exclusive) of range: ')
			incr = raw_input('Enter increment: ')
			epsilon_range = np.arange(float(start), float(end), float(incr))
			avg_train_perf = []
			avg_val_perf = []

			for e in epsilon_range:
				train_perf = []
				val_perf = []

				# Use cross validation to tune parameter
				kf = KFold() # TODO: Use StratifiedKFold
				for train, val in kf.split(train_data):
					X_train, X_val = train_data[train], train_data[val]
					y_train, y_val = outcomes[train].A1, outcomes[val].A1

					model = PassiveAggressiveRegressor(
						epsilon=e,
						random_state=np.random.RandomState(seed)
						)

					for i in xrange(0, len(X_train), mini_batch_size):
						model = model.partial_fit(
							X_train[i:i + mini_batch_size],
							y_train[i:i + mini_batch_size]
							)
					y_pred = model.predict(X_train)
					train_perf.append(regression_performance(
						y_train,
						y_pred,
						'mse'
						))
					y_pred = model.predict(X_val)
					val_perf.append(regression_performance(
						y_val,
						y_pred,
						'mse'
						))
				avg_train_perf.append(
					np.sum(train_perf) * 1.0 / len(train_perf)
					)
				avg_val_perf.append(
					np.sum(val_perf) * 1.0 / len(val_perf)
					)
			plot_hyperparameter(
				epsilon_range,
				avg_train_perf,
				avg_val_perf,
				**{
					'parameter' : r'Epsilon $\epsilon$',
					'score' : 'Mean Squared Error'
				})
		elif comp == 'q':
			return
		else:
			break

	best_epsilon = 0.1 # Default
	epsilon_pick = raw_input('Choose value of epsilon (default: ' +
							 str(best_epsilon) + '): ')
	if epsilon_pick == 'q':
		return
	if epsilon_pick != '':
		best_epsilon = float(epsilon_pick)


	# n_iter
	best_n_iter = 5 # Default
	n_iter = raw_input('Enter the number of epochs (default: ' +
						str(best_n_iter) + '): ')
	if n_iter == 'q':
		return
	if n_iter != '':
		best_n_iter = int(n_iter)


	# shuffle
	shuffle = True # Default
	shuffle_pick = raw_input('Shuffle after each epoch (default: Y)? [Y/n] ')
	if shuffle_pick == 'q':
		return
	if shuffle_pick == 'n':
		shuffle = False


	# fit_intercept
	intercept = True # Default
	pick_intercept = raw_input('Fit intercept (default: Y)? [Y/n] ')
	if pick_intercept == 'q':
		return
	if pick_intercept == 'n':
		intercept = False


	# loss: 'epsilon_insensitive', 'squared_epsilon_insensitive'
	loss = 'epsilon_insensitive' # Default
	pick_loss = raw_input('Epsilon Insensitive (1) or '
						  'Squared_epsilon_insensitive (2)? ')
	if epsilon_pick == 'q':
		return
	if pick_loss == 2:
		loss = 'squared_epsilon_insensitive'


	overall_train_perf = None
	# Create model with tuned parameters
	np.random.seed(seed)
	model = PassiveAggressiveRegressor(
		C=best_C,
		epsilon=best_epsilon,
		fit_intercept=intercept,
		n_iter=best_n_iter,
		shuffle=shuffle,
		random_state=seed,
		loss=loss,
		)
	# Observe how the model performs with increasingly more data
	current_total_data = 0
	incremental_sizes = []
	for i in xrange(0, total_training_instances, mini_batch_size):
		data = train_data[i:i + mini_batch_size]
		out = outcomes[i:i + mini_batch_size]
		model = model.partial_fit(data, out.A1)
		current_total_data += len(data)
		incremental_sizes.append(current_total_data)

		results['Perfusion Parameter'].append(perfusion_param)
		results['Model'].append('Passive Aggressive')
		results['Patch Radius'].append(patch_radius)
		results['Mini Batch Size'].append(mini_batch_size)
		results['C (Regularization)'].append(best_C)
		results['Epsilon'].append(best_epsilon)
		results['Fit Intercept?'].append(intercept)
		results['Number of Epochs'].append(best_n_iter)
		results['Shuffle?'].append(shuffle)
		results['Random Seed'].append(seed)
		results['Loss Function'].append(loss)
		results['Warm Start?'].append(False)
		results['Total Number of Examples Trained'].append(current_total_data)

		# Compute training performance
		y_pred = model.predict(X[0])
		overall_train_perf = regression_performance(
			y[0].A1,
			y_pred,
			'mse'
			)
		results['Training MSE'].append(overall_train_perf)
		overall_train_perf = regression_performance(
			y[0].A1,
			y_pred,
			'r2-score'
			)
		results['Training R^2 Score'].append(overall_train_perf)

		# Compute test performance using test data
		y_pred = model.predict(X[2])
		test_perf = regression_performance(
			y[2].A1,
			y_pred,
			'mse'
			)
		results['Test MSE'].append(test_perf)
		test_perf = regression_performance(
			y[2].A1,
			y_pred,
			'r2-score'
			)
		results['Test R^2 Score'].append(test_perf)

	record_results(results, attributes)
	plot_incremental_performance(
		incremental_sizes,
		results['Training MSE'],
		results['Test MSE'],
		**{ 'score' : 'Mean Squared Error' }
		)
