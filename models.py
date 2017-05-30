from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from util import regression_performance, plot_hyperparameter, \
				 plot_incremental_performance, record_results
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import os

"""
Steps to evaluate a ML technique:
	1. Get the data!
	2. Visualize the data through plots.
	3. Train the model.
	4. Evaluate the model.
		-- Use cross validation (if validation data is not available)
		   to tune hyperparameters.
		-- Look out for overfitting and underfitting.
		-- Repeat Step 3 to find the best model.
"""

def run_SGD(X, y, **kwargs):
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
		'Batch Size',
		'Total Number of Examples Trained',
		'Training MSE',
		'Test MSE',
		'Training R^2 Score',
		'Test R^2 Score',
		'Penalty (Regularization)', 
		'Alpha',
		'Epsilon',
		'Fit Intercept?',
		'Shuffle?',
		'Random Seed',
		'Loss Function',
		'Warm Start?',
		'Average'
	]
	best_penalty = 'elasticnet'
	
	results = {}
	for a in attributes:
		results[a] = []

	batch_size = 100 # Default
	size = raw_input('Enter batch size (default: ' +
					str(batch_size) + '): ')
	if size == 'q':
		return
	if size != '':
		batch_size = int(size)

	# Enter a seed in order to reproduce results (even if the shuffle option
	# is not set to True)
	seed = None # Default
	pick_seed = raw_input('Enter seed of random number generator to '
						  'shuffle: ')
	if pick_seed == 'q':
		return
	if pick_seed != '':
		seed = int(pick_seed)

	# Shuffle data
	indices = np.arange(0, total_training_instances)
	np.random.seed(seed)
	np.random.shuffle(indices)
	train_data = np.matrix([np.asarray(X[0])[i] for i in indices])
	outcomes = np.matrix([[y[0].A1[i]] for i in indices])
	
	print 'Tuning regularization penalty with alpha term...'
	while True:
		comp = raw_input('Compare errors for range of alpha values? [Y/n] ')
		if comp == 'Y':
			start = raw_input('Enter lower bound (inclusive) of range: ')
			end = raw_input('Enter upper bound (exclusive) of range: ')
			incr = raw_input('Enter increment: ')
			alpha_range = np.arange(float(start), float(end), float(incr))

			avg_train_perf = []
			avg_val_perf = []

			for a in alpha_range:
				train_perf = []
				val_perf = []

				# Use cross validation to tune parameter
				kf = KFold()
				for train, val in kf.split(train_data):
					X_train, X_val = train_data[train], train_data[val]
					y_train, y_val = outcomes[train].A1, outcomes[val].A1

					model = SGDRegressor(
						penalty=best_penalty,
						alpha = a,
						random_state=np.random.RandomState(seed)
						)

					for i in xrange(0, len(X_train), batch_size):
						model = model.partial_fit(
							X_train[i:i + batch_size],
							y_train[i:i + batch_size]
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
				alpha_range,
				avg_train_perf,
				avg_val_perf,
				**{
					'parameter' : r'Regularization l2 with multiplier $alpha$',
					'score' : 'Mean Squared Error'
				})
		elif comp == 'q':
			return
		else:
			break

	best_alpha = 0.0001 # Default
	alpha_pick = raw_input('Choose value of alpha (default: ' + str(best_alpha) + '): ')
	if alpha_pick != '':
		best_alpha = float(alpha_pick)


	print 'Tuning epsilon (threshold)...'
	while True:
		comp = raw_input('Compare errors for range of epsilon values? [Y/n] ')
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
				kf = KFold()
				for train, val in kf.split(train_data):
					X_train, X_val = train_data[train], train_data[val]
					y_train, y_val = outcomes[train].A1, outcomes[val].A1

					model = SGDRegressor(
						epsilon=e,
						random_state=np.random.RandomState(seed)
						)

					for i in xrange(0, len(X_train), batch_size):
						model = model.partial_fit(
							X_train[i:i + batch_size],
							y_train[i:i + batch_size]
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

	# average
	sgd_average = False # Default
	avg_pick = raw_input('Take average of SGD weights (default: N)? [Y/n] ')
	if avg_pick == 'q':
		return
	if avg_pick == 'Y':
		sgd_average = True

	# loss: 'epsilon_insensitive', 'squared_epsilon_insensitive'
	loss = 'epsilon_insensitive' # Default
	print 'Choose a loss function to be used.'
	print '1: Epsilon Insensitive (PA-I)'
	print '2: Squared Epsilon Insensitive (PA-II)'
	pick_loss = raw_input('Enter value (default: 1): ')
	if epsilon_pick == 'q':
		return
	if pick_loss == 2:
		loss = 'squared_epsilon_insensitive'


	##########################################################################
	# Observe performance of model after each batch has been trained on.
	##########################################################################
	resp = raw_input('See incremental performance? [Y/n] ')
	if resp == 'q':
		return
	if resp == 'Y':
		# Create model with tuned parameters
		np.random.seed(seed)
		model = SGDRegressor(
			penalty=best_penalty,
			alpha=best_alpha,
			epsilon=best_epsilon,
			fit_intercept=intercept,
			n_iter=1, # Not applicable for partial fit
			shuffle=shuffle,
			random_state=seed,
			loss=loss,
			average=sgd_average
			)
		# Observe how the model performs with increasingly more data
		current_total_data = 0
		incremental_sizes = []
		for i in xrange(0, total_training_instances, batch_size):
			data = train_data[i:i + batch_size]
			out = outcomes[i:i + batch_size]
			model = model.partial_fit(data, out.A1)
			current_total_data += len(data)
			incremental_sizes.append(current_total_data)

			results['Perfusion Parameter'].append(perfusion_param)
			results['Model'].append('Stochastic Gradient Descent')
			results['Patch Radius'].append(patch_radius)
			results['Batch Size'].append(batch_size)
			results['Penalty (Regularization)'].append('l2')
			results['Alpha'].append(best_alpha)
			results['Average'].append(sgd_average)
			results['Epsilon'].append(best_epsilon)
			results['Fit Intercept?'].append(intercept)
			results['Shuffle?'].append(shuffle)
			results['Random Seed'].append(seed)
			results['Loss Function'].append(loss)
			results['Warm Start?'].append(False)
			results['Total Number of Examples Trained'].append(
				current_total_data
				)

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

		record_results(results, attributes, **{
			'title': 'incremental results'
			})
		final_result = {}
		for attr in results:
			final_result[attr] = [results[attr][-1]]
		final_result['Epochs'] = [1]
		attributes.append('Epochs')
		record_results(final_result, attributes, **{
			'title': 'final results'
			})

		# Print summary
		print '================'
		print 'SUMMARY'
		print '================'
		print 'Perfusion Parameter: ' + perfusion_param
		print 'Patch Radius       : ' + str(patch_radius)
		print 'Model              : Stochastic Gradient Descent (SGD)'
		print 'Batch Size         : ' + str(batch_size)
		print '----------------'
		print 'Model Parameters'
		print '----------------'
		print 'Penalty(Regulariz.): ' + best_penalty
		print 'Alpha             : ' + str(best_alpha)
		print 'Epsilon           : ' + str(best_epsilon)
		print 'Fit Intercept     : ' + str(intercept)
		print 'Shuffle           : ' + str(shuffle)
		print 'Random Seed       : ' + str(seed)
		print 'Loss Function     : ' + loss
		print 'Warm Start        : ' + str(False)
		print 'Average           : ' + str(sgd_average)
		print '----------------'
		print 'Results'
		print '----------------'
		print 'Number of Epochs                 : 1'
		print ('Final Training Mean Squared Error: ' +
				str(final_result['Training MSE'][0]))
		print ('Final Training R^2 Score         : ' +
				str(final_result['Training R^2 Score'][0]))
		print ('Final Test Mean Squared Error    : ' +
				str(final_result['Test MSE'][0]))
		print ('Final Test R^2 Score             : ' +
				str(final_result['Test R^2 Score'][0]))
		print

		plot_incremental_performance(
			incremental_sizes,
			results['Training MSE'],
			results['Test MSE'],
			**{ 'score' : 'Mean Squared Error' }
			)

	resp = raw_input('See performance with more than one epoch?\n'
					 'Parameters will remain the same. [Y/n] ')
	if resp == 'q':
		return
	if resp == 'Y':
		print 'Tuning number of epochs...'
		while True:
			comp = raw_input('Compare errors for range of epoch values? [Y/n] ')
			if comp == 'Y':
				start = raw_input('Enter lower bound (inclusive) of range: ')
				end = raw_input('Enter upper bound (exclusive) of range: ')
				incr = raw_input('Enter increment: ')
				n_range = np.arange(int(start), int(end), int(incr))
				avg_train_perf = []
				avg_val_perf = []

				for n in n_range:
					train_perf = []
					val_perf = []

					# Use cross validation to tune parameter
					kf = KFold()
					for train, val in kf.split(train_data):
						X_train, X_val = train_data[train], train_data[val]
						y_train, y_val = outcomes[train].A1, outcomes[val].A1

						"""
						n_iter does not apply for partial fitting so just
						manually iterate n times over data
						"""
						model = SGDRegressor(
							random_state=np.random.RandomState(seed)
							)

						for rnd in xrange(n):
							for i in xrange(0, len(X_train), batch_size):
								model = model.partial_fit(
									X_train[i:i + batch_size],
									y_train[i:i + batch_size]
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
					n_range,
					avg_train_perf,
					avg_val_perf,
					**{
						'parameter' : r'Epochs',
						'score' : 'Mean Squared Error'
					})
			elif comp == 'q':
				return
			else:
				break

		best_n_iter = 5 # Default
		n_iter = raw_input('Enter the number of epochs (default: ' +
							str(best_n_iter) + '): ')
		if n_iter == 'q':
			return
		if n_iter != '':
			best_n_iter = int(n_iter)

		########################################
		# Train model with multiple epochs
		########################################
		np.random.seed(seed)
		model = SGDRegressor(
			average=sgd_average,
			penalty=best_penalty,
			alpha=best_alpha,
			epsilon=best_epsilon,
			fit_intercept=intercept,
			shuffle=shuffle,
			random_state=seed,
			loss=loss,
			)
		for rnd in xrange(best_n_iter):
			for i in xrange(0, total_training_instances, batch_size):
				data = train_data[i:i + batch_size]
				out = outcomes[i:i + batch_size]
				model = model.partial_fit(data, out.A1)

		# Compute training performance
		y_pred = model.predict(X[0])
		overall_train_perf = regression_performance(
			y[0].A1,
			y_pred,
			'mse'
			)
		final_result['Training MSE'] = [overall_train_perf]
		overall_train_perf = regression_performance(
			y[0].A1,
			y_pred,
			'r2-score'
			)
		final_result['Training R^2 Score'] = [overall_train_perf]

		# Compute test performance using test data
		y_pred = model.predict(X[2])
		test_perf = regression_performance(
			y[2].A1,
			y_pred,
			'mse'
			)
		final_result['Test MSE'] = [test_perf]
		test_perf = regression_performance(
			y[2].A1,
			y_pred,
			'r2-score'
			)
		final_result['Test R^2 Score'] = [test_perf]
		final_result['Epochs'] = [best_n_iter]

		record_results(final_result, attributes, **{
			'title': 'final results'
			})
		# Print summary
		print '================'
		print 'SUMMARY'
		print '================'
		print 'Perfusion Parameter: ' + perfusion_param
		print 'Patch Radius       : ' + str(patch_radius)
		print 'Model              : Stochastic Gradient Descent (SGD)'
		print 'Batch Size         : ' + str(batch_size)
		print '----------------'
		print 'Model Parameters'
		print '----------------'
		print 'Penalty(Regulariz.): ' + best_penalty
		print 'Alpha             : ' + str(best_alpha)
		print 'Epsilon           : ' + str(best_epsilon)
		print 'Fit Intercept     : ' + str(intercept)
		print 'Shuffle           : ' + str(shuffle)
		print 'Random Seed       : ' + str(seed)
		print 'Loss Function     : ' + loss
		print 'Warm Start        : ' + str(False)
		print 'Average           : ' + str(sgd_average)
		print '----------------'
		print 'Results'
		print '----------------'
		print 'Number of Epochs                 : ' + str(best_n_iter)
		print ('Final Training Mean Squared Error: ' +
				str(final_result['Training MSE'][0]))
		print ('Final Training R^2 Score         : ' +
				str(final_result['Training R^2 Score'][0]))
		print ('Final Test Mean Squared Error    : ' +
				str(final_result['Test MSE'][0]))
		print ('Final Test R^2 Score             : ' +
				str(final_result['Test R^2 Score'][0]))

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
		'Batch Size',
		'Total Number of Examples Trained',
		'Training MSE',
		'Test MSE',
		'Training R^2 Score',
		'Test R^2 Score',
		'C (Regularization)', # Aggressiveness
		'Epsilon',
		'Fit Intercept?',
		'Shuffle?',
		'Random Seed',
		'Loss Function',
		'Warm Start?'
	]
	results = {}
	for a in attributes:
		results[a] = []

	##########################################################################
	# Pick parameter values.
	##########################################################################

	batch_size = 100 # Default
	size = raw_input('Enter batch size (default: ' +
					str(batch_size) + '): ')
	if size == 'q':
		return
	if size != '':
		batch_size = int(size)

	# Enter a seed in order to reproduce results (even if the shuffle option
	# is not set to True)
	seed = None # Default
	pick_seed = raw_input('Enter seed of random number generator to '
						  'shuffle: ')
	if pick_seed == 'q':
		return
	if pick_seed != '':
		seed = int(pick_seed)

	# Shuffle data
	indices = np.arange(0, total_training_instances)
	np.random.seed(seed)
	np.random.shuffle(indices)
	train_data = np.matrix([np.asarray(X[0])[i] for i in indices])
	outcomes = np.matrix([[y[0].A1[i]] for i in indices])
	
	print 'Tuning regularization C...'
	while True:
		comp = raw_input('Compare errors for range of C values? [Y/n] ')
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
				kf = KFold()
				for train, val in kf.split(train_data):
					X_train, X_val = train_data[train], train_data[val]
					y_train, y_val = outcomes[train].A1, outcomes[val].A1

					model = PassiveAggressiveRegressor(
						C=c,
						random_state=np.random.RandomState(seed)
						)

					for i in xrange(0, len(X_train), batch_size):
						model = model.partial_fit(
							X_train[i:i + batch_size],
							y_train[i:i + batch_size]
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
				kf = KFold()
				for train, val in kf.split(train_data):
					X_train, X_val = train_data[train], train_data[val]
					y_train, y_val = outcomes[train].A1, outcomes[val].A1

					model = PassiveAggressiveRegressor(
						epsilon=e,
						random_state=np.random.RandomState(seed)
						)

					for i in xrange(0, len(X_train), batch_size):
						model = model.partial_fit(
							X_train[i:i + batch_size],
							y_train[i:i + batch_size]
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
	print 'Choose a loss function to be used.'
	print '1: Epsilon Insensitive (PA-I)'
	print '2: Squared Epsilon Insensitive (PA-II)'
	pick_loss = raw_input('Enter value (default: 1): ')
	if epsilon_pick == 'q':
		return
	if pick_loss == 2:
		loss = 'squared_epsilon_insensitive'


	##########################################################################
	# Observe performance of model for each batch that has been trained on.
	##########################################################################
	resp = raw_input('See incremental performance? [Y/n] ')
	if resp == 'q':
		return
	if resp == 'Y':
		# Create model with tuned parameters
		np.random.seed(seed)
		model = PassiveAggressiveRegressor(
			C=best_C,
			epsilon=best_epsilon,
			fit_intercept=intercept,
			n_iter=1, # Not applicable for partial fit
			shuffle=shuffle,
			random_state=seed,
			loss=loss,
			)
		# Observe how the model performs with increasingly more data
		current_total_data = 0
		incremental_sizes = []
		for i in xrange(0, total_training_instances, batch_size):
			data = train_data[i:i + batch_size]
			out = outcomes[i:i + batch_size]
			model = model.partial_fit(data, out.A1)
			current_total_data += len(data)
			incremental_sizes.append(current_total_data)

			results['Perfusion Parameter'].append(perfusion_param)
			results['Model'].append('PA')
			results['Patch Radius'].append(patch_radius)
			results['Batch Size'].append(batch_size)
			results['C (Regularization)'].append(best_C)
			results['Epsilon'].append(best_epsilon)
			results['Fit Intercept?'].append(intercept)
			results['Shuffle?'].append(shuffle)
			results['Random Seed'].append(seed)
			results['Loss Function'].append(loss)
			results['Warm Start?'].append(False)
			results['Total Number of Examples Trained'].append(
				current_total_data
				)

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

		record_results(results, attributes, **{
			'title': 'incremental results'
			})
		plot_incremental_performance(
			incremental_sizes,
			results['Training MSE'],
			results['Test MSE'],
			**{ 'score' : 'Mean Squared Error' }
			)

		final_result = {}
		for attr in results:
			final_result[attr] = [results[attr][-1]]
		final_result['Epochs'] = [1]
		attributes.append('Epochs')
		record_results(final_result, attributes, **{
			'title': 'final results'
			})

		# Print summary
		print '================'
		print 'SUMMARY'
		print '================'
		print 'Perfusion Parameter: ' + perfusion_param
		print 'Patch Radius       : ' + str(patch_radius)
		print 'Model              : Passive Aggressive (PA)'
		print 'Batch Size         : ' + str(batch_size)
		print '----------------'
		print 'Model Parameters'
		print '----------------'
		print 'C (Regularization): ' + str(best_C)
		print 'Epsilon           : ' + str(best_epsilon)
		print 'Fit Intercept     : ' + str(intercept)
		print 'Shuffle           : ' + str(shuffle)
		print 'Random Seed       : ' + str(seed)
		print 'Loss Function     : ' + loss
		print 'Warm Start        : ' + str(False)
		print '----------------'
		print 'Results'
		print '----------------'
		print 'Number of Epochs                 : 1'
		print ('Final Training Mean Squared Error: ' +
				str(final_result['Training MSE'][0]))
		print ('Final Training R^2 Score         : ' +
				str(final_result['Training R^2 Score'][0]))
		print ('Final Test Mean Squared Error    : ' +
				str(final_result['Test MSE'][0]))
		print ('Final Test R^2 Score             : ' +
				str(final_result['Test R^2 Score'][0]))
		print

	##########################################################################
	# Observe performance of model with more than 1 epoch.
	##########################################################################
	
	resp = raw_input('See performance with more than one epoch?\n'
					 'Parameters will remain the same. [Y/n] ')
	if resp == 'q':
		return
	if resp == 'Y':
		print 'Tuning number of epochs...'

		# Initialize final result object
		final_result = {}
		for a in attributes:
			final_result[a] = []
		final_result['Perfusion Parameter'].append(perfusion_param)
		final_result['Model'].append('PA')
		final_result['Patch Radius'].append(patch_radius)
		final_result['Batch Size'].append(batch_size)
		final_result['C (Regularization)'].append(best_C)
		final_result['Epsilon'].append(best_epsilon)
		final_result['Fit Intercept?'].append(intercept)
		final_result['Shuffle?'].append(shuffle)
		final_result['Random Seed'].append(seed)
		final_result['Loss Function'].append(loss)
		final_result['Warm Start?'].append(False)
		final_result['Total Number of Examples Trained'].append(
			total_training_instances
			)

		while True:
			comp = raw_input('Compare errors for range of epoch values? [Y/n] ')
			if comp == 'Y':
				start = raw_input('Enter lower bound (inclusive) of range: ')
				end = raw_input('Enter upper bound (exclusive) of range: ')
				incr = raw_input('Enter increment: ')
				n_range = np.arange(int(start), int(end), int(incr))
				avg_train_perf = []
				avg_val_perf = []

				for n in n_range:
					train_perf = []
					val_perf = []

					# Use cross validation to tune parameter
					kf = KFold()
					for train, val in kf.split(train_data):
						X_train, X_val = train_data[train], train_data[val]
						y_train, y_val = outcomes[train].A1, outcomes[val].A1

						"""
						n_iter does not apply for partial fitting so just
						manually iterate n times over data
						"""
						model = PassiveAggressiveRegressor(
							random_state=np.random.RandomState(seed)
							)

						for rnd in xrange(n):
							for i in xrange(0, len(X_train), batch_size):
								model = model.partial_fit(
									X_train[i:i + batch_size],
									y_train[i:i + batch_size]
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
					n_range,
					avg_train_perf,
					avg_val_perf,
					**{
						'parameter' : r'Epochs',
						'score' : 'Mean Squared Error'
					})
			elif comp == 'q':
				return
			else:
				break

		best_n_iter = 5 # Default
		n_iter = raw_input('Enter the number of epochs (default: ' +
							str(best_n_iter) + '): ')
		if n_iter == 'q':
			return
		if n_iter != '':
			best_n_iter = int(n_iter)

		######################################################################
		# Train model with multiple epochs
		######################################################################
		np.random.seed(seed)
		model = PassiveAggressiveRegressor(
			C=best_C,
			epsilon=best_epsilon,
			fit_intercept=intercept,
			shuffle=shuffle,
			random_state=seed,
			loss=loss,
			)
		for rnd in xrange(best_n_iter):
			for i in xrange(0, total_training_instances, batch_size):
				data = train_data[i:i + batch_size]
				out = outcomes[i:i + batch_size]
				model = model.partial_fit(data, out.A1)

		# Compute training performance
		y_pred = model.predict(X[0])
		overall_train_perf = regression_performance(
			y[0].A1,
			y_pred,
			'mse'
			)
		final_result['Training MSE'].append(overall_train_perf)
		overall_train_perf = regression_performance(
			y[0].A1,
			y_pred,
			'r2-score'
			)
		final_result['Training R^2 Score'].append(overall_train_perf)

		# Compute test performance using test data
		y_pred = model.predict(X[2])
		test_perf = regression_performance(
			y[2].A1,
			y_pred,
			'mse'
			)
		final_result['Test MSE'].append(test_perf)
		test_perf = regression_performance(
			y[2].A1,
			y_pred,
			'r2-score'
			)
		final_result['Test R^2 Score'].append(test_perf)
		final_result['Epochs'] = [best_n_iter]

		if 'Epochs' not in attributes:
			attributes.append('Epochs')
		record_results(final_result, attributes, **{
			'title': 'final results'
			})
		# Print summary
		print '================'
		print 'SUMMARY'
		print '================'
		print 'Perfusion Parameter: ' + perfusion_param
		print 'Patch Radius       : ' + str(patch_radius)
		print 'Model              : Passive Aggressive (PA)'
		print 'Batch Size         : ' + str(batch_size)
		print '----------------'
		print 'Model Parameters'
		print '----------------'
		print 'C (Regularization): ' + str(best_C)
		print 'Epsilon           : ' + str(best_epsilon)
		print 'Fit Intercept     : ' + str(intercept)
		print 'Shuffle           : ' + str(shuffle)
		print 'Random Seed       : ' + str(seed)
		print 'Loss Function     : ' + loss
		print 'Warm Start        : ' + str(False)
		print '----------------'
		print 'Results'
		print '----------------'
		print 'Number of Epochs                 : ' + str(best_n_iter)
		print ('Final Training Mean Squared Error: ' +
				str(final_result['Training MSE'][0]))
		print ('Final Training R^2 Score         : ' +
				str(final_result['Training R^2 Score'][0]))
		print ('Final Test Mean Squared Error    : ' +
				str(final_result['Test MSE'][0]))
		print ('Final Test R^2 Score             : ' +
				str(final_result['Test R^2 Score'][0]))
