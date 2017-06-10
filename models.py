from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from util import regression_performance, plot_hyperparameter, \
				 learning_curve, record_results
from sklearn.model_selection import KFold
from controls import print_controls, ctrls
import numpy as np
import pandas as pd
import os
import copy
import traceback

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

def run_multiple_SGD(X, y, **kwargs):
	"""
	Runs Stochastic Gradient Descent algorithm multiple times
		using different seeds on the regression data.

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

	total_training_instances = len(X[0])
	controls = ctrls()

	if 'parameter' not in kwargs:
		perfusion_param = None
	else:
		perfusion_param = kwargs.pop('parameter')

	if 'patch_radius' not in kwargs:
		patch_radius = None
	else:
		patch_radius = kwargs.pop('patch_radius')

	seeds = raw_input('Enter seeds (separated by space): ')
	seeds = [int(s) for s in seeds.split()]

	total_training_instances = len(X[0])
	batch_size = 100
	
	# Shuffle data
	indices = np.arange(0, total_training_instances)
	np.random.seed(seeds[0])
	np.random.shuffle(indices)
	train_data = np.matrix([np.asarray(X[0])[i] for i in indices])
	outcomes = np.matrix([[y[0].A1[i]] for i in indices])
	

	



	best_penalty = 'l2' # 
	print 'Choose a penalty (regularization term) to be used.'
	print '1: none'
	print '2: l2'
	print '3: l1'
	print '4: elasticnet'
	pick_penalty = raw_input('Enter value (default: 2): ')
	if pick_penalty == controls['Quit']:
		return
	if pick_penalty == '1':
		best_penalty = 'none'
	elif pick_penalty == '3':
		best_penalty = 'l1'
	elif pick_penalty == '4':
		best_penalty = 'elasticnet'
	
	
	print 'Enter range of alpha term (default 0.0001) for penalty ' + best_penalty
	start = raw_input('\tEnter lower bound (inclusive) of range: ')
	end = raw_input('\tEnter upper bound (exclusive) of range: ')
	incr = raw_input('\tEnter increment: ')
	alpha_range = np.arange(float(start), float(end), float(incr))
	min_test_err = None
	best_alpha = None
	for a in alpha_range:
		avg_test_errs = np.array([])
		print "Testing alpha " + str(a)
		for seed in seeds[1:]:
			model = SGDRegressor(
					penalty=best_penalty,
					alpha = a,
					random_state=np.random.RandomState(seed)
					)
			for i in xrange(0, total_training_instances, batch_size):
				data = train_data[i:i+batch_size]
				out = outcomes[i:i+batch_size]
				model = model.partial_fit(data, out.A1)						
				
			y_pred = model.predict(X[2])
			avg_test_errs = np.append(avg_test_errs,[
				regression_performance(y[2].A1,y_pred,'rms')
				])
		err = avg_test_errs.mean()
		if min_test_err is None or err < min_test_err:
			min_test_err = err
			best_alpha = a
				

			
	best_learn = 'optimal' # Default
	# 'none', 'l2', 'l1', or 'elasticnet'
	print 'Choose a learning rate schedule to be used.'
	print '1: Constant'
	print '2: Optimal'
	print '3: Inverse Scaling'
	pick_learn = raw_input('Enter value (default: 2): ')
	if pick_learn == controls['Quit']:
		return
	if pick_learn == '1':
		best_learn = 'constant'
	elif pick_learn == '3':
		best_learn = 'invscaling'
		
	print 'Enter range of initial learning rate (default 0.01) '
	start = raw_input('\tEnter lower bound (inclusive) of range: ')
	end = raw_input('\tEnter upper bound (exclusive) of range: ')
	incr = raw_input('\tEnter increment: ')
	learn_range = np.arange(float(start), float(end), float(incr))
	min_test_err = None
	best_learnRate = None
	for a in learn_range:
		avg_test_errs = np.array([])
		print "Testing initial rate" + str(a)
		for seed in seeds[1:]:
			model = SGDRegressor(
					learning_rate=best_learn,
					eta0 = a,
					random_state=np.random.RandomState(seed)
					)
			for i in xrange(0, total_training_instances, batch_size):
				data = train_data[i:i+batch_size]
				out = outcomes[i:i+batch_size]
				model = model.partial_fit(data, out.A1)						
				
			y_pred = model.predict(X[2])
			avg_test_errs = np.append(avg_test_errs,[
				regression_performance(
				y[2].A1,
				y_pred,
				'rms'
				)])
		err = avg_test_errs.mean()
		if min_test_err is None or err < min_test_err:
			min_test_err = err
			best_learnRate = a
			

	
	best_powT = 0.25
	if best_learn == 'invscaling':
		print 'Enter range of exponent for inverse scaling (default 0.25) '
		start = raw_input('\tEnter lower bound (inclusive) of range: ')
		end = raw_input('\tEnter upper bound (exclusive) of range: ')
		incr = raw_input('\tEnter increment: ')
		exp_range = np.arange(float(start), float(end), float(incr))
		min_test_err = None
		best_powT = None
		for a in exp_range:
			print "Testing exponent " + str(a)
			avg_test_errs = np.array([])
			for seed in seeds[1:]:
				model = SGDRegressor(
						learning_rate=best_learn,
						eta0 = a,
						power_t = a,
						random_state=np.random.RandomState(seed)
						)
				for i in xrange(0, total_training_instances, batch_size):
					data = train_data[i:i+batch_size]
					out = outcomes[i:i+batch_size]
					model = model.partial_fit(data, out.A1)						
					
				y_pred = model.predict(X[2])
				avg_test_errs = np.append(avg_test_errs,[
					regression_performance(
					y[2].A1,
					y_pred,
					'rms'
					)])
			err = avg_test_errs.mean()
			if min_test_err is None or err < min_test_err:
				min_test_err = err
				best_powT = a
	
	# default = 0.1
	print 'Enter range of epsilon.'
	start = raw_input('\tEnter lower bound (inclusive) of range: ')
	end = raw_input('\tEnter upper bound (exclusive) of range: ')
	incr = raw_input('\tEnter increment: ')
	epsilon_range = np.arange(float(start), float(end), float(incr))
	min_test_err = None
	best_epsilon = None
	for e in epsilon_range:
		avg_test_errs = np.array([])
		print "Testing epsilon value " + str(e)
		for seed in seeds[1:]:
			model = SGDRegressor(
				epsilon=e,
				random_state=np.random.RandomState(seed)
				)
			for i in xrange(0, total_training_instances, batch_size):
				data = train_data[i:i + batch_size]
				out = outcomes[i:i + batch_size]
				model = model.partial_fit(data, out.A1)

			y_pred = model.predict(X[2])
			avg_test_errs = np.append(avg_test_errs, [
				regression_performance(y[2].A1, y_pred, 'rms')
				])
		err = avg_test_errs.mean()
		if min_test_err is None or err < min_test_err:
			min_test_err = err
			best_epsilon = e

	print 'Enter range of epochs.'
	start = raw_input('\tEnter lower bound (inclusive) of range: ')
	end = raw_input('\tEnter upper bound (exclusive) of range: ')
	incr = raw_input('\tEnter increment: ')
	epoch_range = np.arange(int(start), int(end), int(incr))
	min_test_err = None
	best_epochs = None
	for e in epoch_range:
		avg_test_errs = np.array([])
		print "Testing epoch number " + str(e)
		for seed in seeds[1:]:
			model = PassiveAggressiveRegressor(
				random_state=np.random.RandomState(seed)
				)
			for rnd in xrange(e):
				for i in xrange(0, total_training_instances, batch_size):
					data = train_data[i:i + batch_size]
					out = outcomes[i:i + batch_size]
					model = model.partial_fit(data, out.A1)

			y_pred = model.predict(X[2])
			avg_test_errs = np.append(avg_test_errs, [
				regression_performance(y[2].A1, y_pred, 'rms')
				])
		err = avg_test_errs.mean()
		if min_test_err is None or err < min_test_err:
			min_test_err = err
			best_epochs = e

		
	attributes = [
		'Perfusion Parameter',
		'Model',
		'Patch Radius',
		'Batch Size',
		'Total Number of Examples Trained',
		'Trial',
		'Training RMSE',
		'Test RMSE',
		'Training R^2 Score',
		'Test R^2 Score',
		'Penalty (Regularization)', 
		'Alpha',
		'Epsilon',
		'Learning Rate',
		'eta0', # The initial learning rate [default 0.01].
		'Exponent (inv scaling)', # The exponent for inverse scaling learning rate [default 0.25].
		'Epochs',
		'Fit Intercept?',
		'Shuffle?',
		'Loss Function',
		'Warm Start?',
		'Average'
	] ## might add/ change one of these attributes
	
	# Pick values for parameters
	results = {}
	for a in attributes:
		results[a] = []
		

	trials = raw_input('Enter number of trials: ')
	trials = int(trials)

	indices = np.arange(0, total_training_instances)
	np.random.shuffle(indices)
	train_data = np.matrix([np.asarray(X[0])[i] for i in indices])
	outcomes = np.matrix([[y[0].A1[i]] for i in indices])

	np.random.seed(None)
	print "Begin multiple trials test"
	# Create model with tuned parameters
	prev = -1
	for trial in xrange(trials):
		if int(((trial*100.0)/trials)) % 10 == 0 and int(((trial*100.0)/trials)) != prev: 
			#display a note every 10% so user knows program didn't freeze
				prev = int(((trial*100.0)/trials))
				print "Trials " + str(prev) + "% complete"
		model = SGDRegressor(
			penalty=best_penalty,
			alpha=best_alpha,
			epsilon=best_epsilon,
			fit_intercept=True,
			n_iter=1, # Not applicable for partial fit
			shuffle=True,
			random_state=None,
			loss='epsilon_insensitive',
			average=False,
			learning_rate=best_learn,
			eta0 = best_learnRate,
			power_t=best_powT
			)
		for rnd in xrange(best_epochs):
			for i in xrange(0, total_training_instances, batch_size):
				data = train_data[i:i + batch_size]
				out = outcomes[i:i + batch_size]
				model = model.partial_fit(data, out.A1)
				
		results['Perfusion Parameter'].append(perfusion_param)
		results['Model'].append('SGD')
		results['Patch Radius'].append(patch_radius)
		results['Batch Size'].append(batch_size)
		results['Penalty (Regularization)'].append(best_penalty)
		results['Alpha'].append(best_alpha)
		results['Average'].append(False)
		results['Epsilon'].append(best_epsilon)
		results['Fit Intercept?'].append(True)
		results['Shuffle?'].append(True)
		results['Epochs'].append(best_epochs)
		results['Loss Function'].append('epsilon_insensitive')
		results['Warm Start?'].append(False)
		results['Learning Rate'].append(best_learn)
		results['Trial'].append(trial + 1)
		results['eta0'].append(best_learnRate)
		results['Exponent (inv scaling)'].append(best_powT)
		results['Total Number of Examples Trained'].append(
				total_training_instances
				)

		# Compute training performance
		y_pred = model.predict(X[0])
		overall_train_perf = regression_performance(y[0].A1,y_pred,'rms')
		
		results['Training RMSE'].append(overall_train_perf)
		overall_train_perf = regression_performance(y[0].A1,y_pred,'r2-score')
		results['Training R^2 Score'].append(overall_train_perf)

		# Compute test performance using test data
		y_pred = model.predict(X[2])
		test_perf = regression_performance(y[2].A1,y_pred,'rms')
		results['Test RMSE'].append(test_perf)
		test_perf = regression_performance(y[2].A1,y_pred,'r2-score')
		results['Test R^2 Score'].append(test_perf)
		
	record_results(results, attributes, **{
		'title': 'trial results'
		})


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
	controls = ctrls()

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
		'Training RMSE',
		'Test RMSE',
		'Training R^2 Score',
		'Test R^2 Score',
		'Penalty (Regularization)', 
		'Alpha',
		'Epsilon',
		'Learning Rate',
		'eta0', # The initial learning rate [default 0.01].
		'Exponent (inv scaling)', # The exponent for inverse scaling learning rate [default 0.25].
		'Fit Intercept?',
		'Shuffle?',
		'Random Seed',
		'Loss Function',
		'Warm Start?',
		'Average'
	] 
	
	
	# Pick values for parameters
	results = {}
	for a in attributes:
		results[a] = []

	batch_size = 100 # Default
	size = raw_input('Enter batch size (default: ' +
					str(batch_size) + '): ')
	if size == controls['Quit']:
		return
	if size != '':
		batch_size = int(size)

	# Enter a seed in order to reproduce results (even if the shuffle option
	# is not set to True)
	seed = None # Default
	pick_seed = raw_input('Enter seed of random number generator to '
						  'shuffle: ')
	if pick_seed == controls['Quit']:
		return
	if pick_seed != '':
		seed = int(pick_seed)

	# Shuffle data
	indices = np.arange(0, total_training_instances)
	np.random.seed(seed)
	np.random.shuffle(indices)
	train_data = np.matrix([np.asarray(X[0])[i] for i in indices])
	outcomes = np.matrix([[y[0].A1[i]] for i in indices])
	
	best_penalty = 'l2' # Default
	# 'none', 'l2', 'l1', or 'elasticnet'
	print 'Choose a penalty (regularization term) to be used.'
	print '1: none'
	print '2: l2'
	print '3: l1'
	print '4: elasticnet'
	pick_penalty = raw_input('Enter value (default: 2): ')
	if pick_penalty == controls['Quit']:
		return
	if pick_penalty == '1':
		best_penalty = 'none'
	elif pick_penalty == '3':
		best_penalty = 'l1'
	elif pick_penalty == '4':
		best_penalty = 'elasticnet'
	
		
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
						'rms'
						))
					y_pred = model.predict(X_val)
					val_perf.append(regression_performance(
						y_val,
						y_pred,
						'rms'
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
					'score' : 'Root Mean Squared Error'
				})
		elif comp == controls['Quit']:
			return
		else:
			break
		
			
	best_alpha = 0.0001 # Default
	alpha_pick = raw_input('Choose value of alpha (default: ' + str(best_alpha) + '): ')
	if alpha_pick != '':
		best_alpha = float(alpha_pick)

	best_learn = 'optimal' # Default
	# 'none', 'l2', 'l1', or 'elasticnet'
	print 'Choose a learning rate schedule to be used.'
	print '1: Constant'
	print '2: Optimal'
	print '3: Inverse Scaling'
	pick_learn = raw_input('Enter value (default: 2): ')
	if pick_learn == controls['Quit']:
		return
	if pick_learn == '1':
		best_learn = 'constant'
	elif pick_learn == '3':
		best_learn = 'invscaling'
		
	print 'Tuning initial learning rate for ' +  best_learn + '...'
	while True:
		comp = raw_input('Compare errors for range of initial learning rates (default 0.01)? [Y/n] ')
		if comp == 'Y':
			start = raw_input('Enter lower bound (inclusive) of range: ')
			end = raw_input('Enter upper bound (exclusive) of range: ')
			incr = raw_input('Enter increment: ')
			learnRate_range = np.arange(float(start), float(end), float(incr))

			avg_train_perf = []
			avg_val_perf = []

			for a in learnRate_range:
				train_perf = []
				val_perf = []
				print "Testing initial learn rate " + str(a)
				# Use cross validation to tune parameter
				kf = KFold()
				for train, val in kf.split(train_data):
					X_train, X_val = train_data[train], train_data[val]
					y_train, y_val = outcomes[train].A1, outcomes[val].A1

					model = SGDRegressor(
						penalty=best_penalty,
						alpha = best_alpha,
						learning_rate=best_learn,
						eta0 = a,
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
						'rms'
						))
					y_pred = model.predict(X_val)
					val_perf.append(regression_performance(
						y_val,
						y_pred,
						'rms'
						))
				avg_train_perf.append(
					np.sum(train_perf) * 1.0 / len(train_perf)
					)
				avg_val_perf.append(
					np.sum(val_perf) * 1.0 / len(val_perf)
					)
			plot_hyperparameter(
				learnRate_range,
				avg_train_perf,
				avg_val_perf,
				**{
					'parameter' : r'Learning Rate type with initial learning rate $eta0$',
					'score' : 'Root Mean Squared Error'
				})
		elif comp == controls['Quit']:
			return
		else:
			break

	best_learnRate = 0.01 # Default
	learnrate_PICK = raw_input('Choose value of initial learn rate for ' +  best_learn + '(default: ' +
							 str(best_learnRate) + '): ')
	if learnrate_PICK == controls['Quit']:
		return
	if learnrate_PICK != '':
		best_learnRate = float(learnrate_PICK)
	
	best_powT = 0.25
	if best_learn == 'invscaling':
		print 'Tuning exponent for inverse scaling learning rate...'
		while True:
			comp = raw_input('Compare errors for range of exponents for inv scaling? [Y/n] ')
			if comp == 'Y':
				start = raw_input('Enter lower bound (inclusive) of range: ')
				end = raw_input('Enter upper bound (exclusive) of range: ')
				incr = raw_input('Enter increment: ')
				exponent_range = np.arange(float(start), float(end), float(incr))

				avg_train_perf = []
				avg_val_perf = []

				for a in exponent_range:
					train_perf = []
					val_perf = []
					#print "Testing inv scaling exponent: " + str(a)
					# Use cross validation to tune parameter
					kf = KFold()
					for train, val in kf.split(train_data):
						X_train, X_val = train_data[train], train_data[val]
						y_train, y_val = outcomes[train].A1, outcomes[val].A1

						model = SGDRegressor(
							penalty=best_penalty,
							alpha = best_alpha,
							learning_rate=best_learn,
							eta0 = best_learnRate,
							power_t=a,
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
							'rms'
							))
						y_pred = model.predict(X_val)
						val_perf.append(regression_performance(
							y_val,
							y_pred,
							'rms'
							))
					avg_train_perf.append(
						np.sum(train_perf) * 1.0 / len(train_perf)
						)
					avg_val_perf.append(
						np.sum(val_perf) * 1.0 / len(val_perf)
						)
				plot_hyperparameter(
					exponent_range,
					avg_train_perf,
					avg_val_perf,
					**{
						'parameter' : r'Inv Scaling Learn with exponent $pow_t$',
						'score' : 'Root Mean Squared Error'
					})
			elif comp == controls['Quit']:
				return
			else:
				break

		powT_PICK = raw_input('Choose value of exponent for inv scaling learn(default: ' +
								 str(best_powT) + '): ')
		if powT_PICK == controls['Quit']:
			return
		if powT_PICK != '':
			best_powT = float(powT_PICK)
			
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
				#print e

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
						'rms'
						))
					y_pred = model.predict(X_val)
					val_perf.append(regression_performance(
						y_val,
						y_pred,
						'rms'
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
					'score' : 'Root Mean Squared Error'
				})
		elif comp == controls['Quit']:
			return
		else:
			break

	best_epsilon = 0.1 # Default
	epsilon_pick = raw_input('Choose value of epsilon (default: ' +
							 str(best_epsilon) + '): ')
	if epsilon_pick == controls['Quit']:
		return
	if epsilon_pick != '':
		best_epsilon = float(epsilon_pick)


	# shuffle
	shuffle = True # Default
	shuffle_pick = raw_input('Shuffle after each epoch (default: Y)? [Y/n] ')
	if shuffle_pick == controls['Quit']:
		return
	if shuffle_pick == 'n':
		shuffle = False


	# fit_intercept
	intercept = True # Default
	pick_intercept = raw_input('Fit intercept (default: Y)? [Y/n] ')
	if pick_intercept == controls['Quit']:
		return
	if pick_intercept == 'n':
		intercept = False

	# average
	sgd_average = False # Default
	avg_pick = raw_input('Take average of SGD weights (default: N)? [Y/n] ')
	if avg_pick == controls['Quit']:
		return
	if avg_pick == 'Y':
		sgd_average = True

	# loss: 'squared_loss', 'huber', 'epsilon_insensitive', or 'squared_epsilon_insensitive'
	loss = 'squared_loss' # Default
	print 'Choose a loss function to be used.'
	print '1: Squared Loss'
	print '2: Huber'
	print '3: Epsilon Insensitive'
	print '4: Squared Epsilon Insensitive'
	pick_loss = raw_input('Enter value (default: 1): ')
	if pick_loss == controls['Quit']:
		return
	elif pick_loss == '2':
		loss = 'huber'
	elif pick_loss == '3':
		loss = 'epsilon_insensitive'
	elif pick_loss == '4':
		loss = 'squared_epsilon_insensitive'


	##########################################################################
	# Observe performance of model for each batch that has been trained on.
	##########################################################################
	resp = raw_input('See incremental performance? [Y/n] ')
	if resp == controls['Quit']:
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
			average=sgd_average,
			learning_rate=best_learn,
			eta0 = best_learnRate,
			power_t=best_powT
			)
		# Observe how the model performs with increasingly more data
		current_total_data = 0
		incremental_sizes = []
		prev = -1
		for i in xrange(0, total_training_instances, batch_size):
			#print i
			if int(((i*100.0)/total_training_instances)) % 10 == 0 and int(((i*100.0)/total_training_instances)) != prev: 
			#display a note every 10% so user knows program didn't freeze
				prev = int(((i*100.0)/total_training_instances))
				print str(prev) + "% complete"
			data = train_data[i:i + batch_size]
			out = outcomes[i:i + batch_size]
			model = model.partial_fit(data, out.A1)
			current_total_data += len(data)
			incremental_sizes.append(current_total_data)

			results['Perfusion Parameter'].append(perfusion_param)
			results['Model'].append('SGD')
			results['Patch Radius'].append(patch_radius)
			results['Batch Size'].append(batch_size)
			results['Penalty (Regularization)'].append(best_penalty)
			results['Alpha'].append(best_alpha)
			results['Average'].append(sgd_average)
			results['Epsilon'].append(best_epsilon)
			results['Fit Intercept?'].append(intercept)
			results['Shuffle?'].append(shuffle)
			results['Random Seed'].append(seed)
			results['Loss Function'].append(loss)
			results['Warm Start?'].append(False)
			results['Learning Rate'].append(best_learn)
			results['eta0'].append(best_learnRate)
			results['Exponent (inv scaling)'].append(best_powT)
			results['Total Number of Examples Trained'].append(
				current_total_data
				)

			# Compute training performance
			y_pred = model.predict(X[0])
			overall_train_perf = regression_performance(
				y[0].A1,
				y_pred,
				'rms'
				)
			results['Training RMSE'].append(overall_train_perf)
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
				'rms'
				)
			results['Test RMSE'].append(test_perf)
			test_perf = regression_performance(
				y[2].A1,
				y_pred,
				'r2-score'
				)
			results['Test R^2 Score'].append(test_perf)

		
		record_results(results, attributes, **{
			'title': 'incremental results'
			})
			
		learning_curve(
			incremental_sizes,
			results['Training RMSE'],
			results['Test RMSE'],
			**{ 'score' : 'Root Mean Squared Error' }
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
		print 'Model              : Stochastic Gradient Descent (SGD)'
		print 'Batch Size         : ' + str(batch_size)
		print '----------------'
		print 'Model Parameters'
		print '----------------'
		print 'Penalty(Regulariz.): ' + best_penalty
		print 'Alpha              : ' + str(best_alpha)
		print 'Epsilon            : ' + str(best_epsilon)
		print 'Fit Intercept      : ' + str(intercept)
		print 'Shuffle            : ' + str(shuffle)
		print 'Random Seed        : ' + str(seed)
		print 'Loss Function      : ' + loss
		print 'Warm Start         : ' + str(False)
		print 'Average            : ' + str(sgd_average)
		print 'Learning Rate Sched: ' + str(best_learn)
		print 'Initial Learn Rate : ' + str(best_learnRate)
		print 'Exponent (inv scale):' + str(best_powT)
		print '----------------'
		print 'Results'
		print '----------------'
		print 'Number of Epochs                      : 1'
		print ('Final Training Root Mean Squared Error: ' +
				str(final_result['Training RMSE'][0]))
		print ('Final Training R^2 Score              : ' +
				str(final_result['Training R^2 Score'][0]))
		print ('Final Test Root Mean Squared Error    : ' +
				str(final_result['Test RMSE'][0]))
		print ('Final Test R^2 Score                  : ' +
				str(final_result['Test R^2 Score'][0]))
		print

		
		# Observe performance of model with more than 1 epoch.
		
	resp = raw_input('See performance with more than one epoch?\n'
					 'Parameters will remain the same. [Y/n] ')
	if resp == controls['Quit']:
		return
	if resp == 'Y':
		print 'Tuning number of epochs...'
		final_result = {}
		for a in attributes:
			final_result[a] = []
		
		final_result['Perfusion Parameter'].append(perfusion_param)
		final_result['Model'].append('SGD')
		final_result['Patch Radius'].append(patch_radius)
		final_result['Batch Size'].append(batch_size)
		final_result['Penalty (Regularization)'].append(best_penalty)
		final_result['Alpha'].append(best_alpha)
		final_result['Average'].append(sgd_average)
		final_result['Epsilon'].append(best_epsilon)
		final_result['Fit Intercept?'].append(intercept)
		final_result['Shuffle?'].append(shuffle)
		final_result['Random Seed'].append(seed)
		final_result['Loss Function'].append(loss)
		final_result['Warm Start?'].append(False)
		final_result['Learning Rate'].append(best_learn)
		final_result['eta0'].append(best_learnRate)
		final_result['Exponent (inv scaling)'].append(best_powT)
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

				np.random.seed(seed)
				for n in n_range:
					train_perf = []
					val_perf = []
				
					if shuffle:
						np.random.shuffle(indices)
						train_data = np.matrix([np.asarray(X[0])[i] for i in indices])
						outcomes = np.matrix([[y[0].A1[i]] for i in indices])
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
							'rms'
							))
						y_pred = model.predict(X_val)
						val_perf.append(regression_performance(
							y_val,
							y_pred,
							'rms'
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
						'score' : 'Root Mean Squared Error'
					})
			elif comp == controls['Quit']:
				return
			else:
				break

		best_n_iter = 5 # Default
		n_iter = raw_input('Enter the number of epochs (default: ' +
							str(best_n_iter) + '): ')
		if n_iter == controls['Quit']:
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
			learning_rate=best_learn,
			eta0=best_learnRate,
			power_t=best_powT
			)
		for rnd in xrange(best_n_iter):
			if shuffle:
				np.random.shuffle(indices)
				train_data = np.matrix([np.asarray(X[0])[i] for i in indices])
				outcomes = np.matrix([[y[0].A1[i]] for i in indices])
			for i in xrange(0, total_training_instances, batch_size):
				data = train_data[i:i + batch_size]
				out = outcomes[i:i + batch_size]
				model = model.partial_fit(data, out.A1)

		# Compute training performance
		y_pred = model.predict(X[0])
		overall_train_perf = regression_performance(
			y[0].A1,
			y_pred,
			'rms'
			)
		final_result['Training RMSE'].append(overall_train_perf)
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
			'rms'
			)
		final_result['Test RMSE'].append(test_perf)
		test_perf = regression_performance(
			y[2].A1,
			y_pred,
			'r2-score'
			)
		final_result['Test R^2 Score'].append(test_perf)
		final_result['Epochs'] = [best_n_iter]
		
		if 'Epochs' not in attributes:
			attributes.append('Epochs')
		
		
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
		print 'Alpha              : ' + str(best_alpha)
		print 'Epsilon            : ' + str(best_epsilon)
		print 'Fit Intercept      : ' + str(intercept)
		print 'Shuffle            : ' + str(shuffle)
		print 'Random Seed        : ' + str(seed)
		print 'Loss Function      : ' + loss
		print 'Warm Start         : ' + str(False)
		print 'Average            : ' + str(sgd_average)
		print 'Learning Rate Sched: ' + str(best_learn)
		print 'Initial Learn Rate : ' + str(best_learnRate)
		print 'Exponent (inv scale):' + str(best_powT)
		print '----------------'
		print 'Results'
		print '----------------'
		print 'Number of Epochs                 : ' + str(best_n_iter)
		print ('Final Training Root Mean Squared Error: ' +
				str(final_result['Training RMSE'][0]))
		print ('Final Training R^2 Score         : ' +
				str(final_result['Training R^2 Score'][0]))
		print ('Final Test Mean Squared Error    : ' +
				str(final_result['Test RMSE'][0]))
		print ('Final Test R^2 Score             : ' +
				str(final_result['Test R^2 Score'][0]))
		
		record_results(final_result, attributes, **{
			'title': 'final results'
			})
	print 'Done'


def home_SGD(X, y, **kwargs):
	"""
	Requests an action from the user to run the Stochastic Gradient
	Descent	algorithm on the regression data.

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

	controls = ctrls()
	suc = True
	while True:
		if suc:
			print '##############'
			print '## SGD Home ##'
			print '##############'
		print 'How would you like to run Stochastic Gradient Descent(SGD)?'
		print '1. Run manually.'
		print '2. Run multiple times to get an average performance.'
		print 'Quit to run on new data.'
		pa_op = raw_input('Enter value: ')
		if pa_op == controls['Help']:
			print_controls()
			suc = False
		elif pa_op == controls['Quit']:
			return
		elif pa_op == controls['Skip']:
			print ('Unable to skip. Press ' + controls['Quit'] +
				   ' to exit.')
			suc = False
		elif pa_op == controls['Home']:
			suc = False
		else:
			try:
				pa_op = int(pa_op)
				if pa_op not in [1, 2]:
					print 'Invalid value. Try again.'
				else:
					if pa_op == 1:
						run_SGD(X, y, **kwargs)
						suc = True
					elif pa_op == 2:
						run_multiple_SGD(X, y, **kwargs)
						suc = True
			except ValueError:
				suc = False
				traceback.print_exc()
				print 'Invalid value. Try again.'
		print

	
def run_multiple_PA(X, y, **kwargs):
	"""
	Runs the Passive-Aggressive algorithm multiple times, using different
	seeds, on the regression data.

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

	if 'parameter' not in kwargs:
		perfusion_param = None
	else:
		perfusion_param = kwargs.pop('parameter')

	if 'patch_radius' not in kwargs:
		patch_radius = None
	else:
		patch_radius = kwargs.pop('patch_radius')

	seeds = raw_input('Enter seeds (separated by space): ')
	seeds = [int(s) for s in seeds.split()]

	total_training_instances = len(X[0])
	batch_size = 100

	# Shuffle data
	indices = np.arange(0, total_training_instances)
	np.random.seed(seeds[0])
	np.random.shuffle(indices)
	train_data = np.matrix([np.asarray(X[0])[i] for i in indices])
	outcomes = np.matrix([[y[0].A1[i]] for i in indices])

	print 'Enter range of C.'
	start = raw_input('\tEnter lower bound (inclusive) of range: ')
	end = raw_input('\tEnter upper bound (exclusive) of range: ')
	incr = raw_input('\tEnter increment: ')
	C_range = np.arange(float(start), float(end), float(incr))
	min_test_err = None
	best_C = None
	for c in C_range:
		avg_test_errs = np.array([])
		for seed in seeds[1:]:
			model = PassiveAggressiveRegressor(
				C=c,
				random_state=np.random.RandomState(seed)
				)
			for i in xrange(0, total_training_instances, batch_size):
				data = train_data[i:i + batch_size]
				out = outcomes[i:i + batch_size]
				model = model.partial_fit(data, out.A1)

			y_pred = model.predict(X[2])
			avg_test_errs = np.append(avg_test_errs, [
				regression_performance(y[2].A1, y_pred, 'rms')
				])
		err = avg_test_errs.mean()
		if min_test_err is None or err < min_test_err:
			min_test_err = err
			best_C = c

	print 'Enter range of epsilon.'
	start = raw_input('\tEnter lower bound (inclusive) of range: ')
	end = raw_input('\tEnter upper bound (exclusive) of range: ')
	incr = raw_input('\tEnter increment: ')
	epsilon_range = np.arange(float(start), float(end), float(incr))
	min_test_err = None
	best_epsilon = None
	for e in epsilon_range:
		avg_test_errs = np.array([])
		for seed in seeds[1:]:
			model = PassiveAggressiveRegressor(
				epsilon=e,
				random_state=np.random.RandomState(seed)
				)
			for i in xrange(0, total_training_instances, batch_size):
				data = train_data[i:i + batch_size]
				out = outcomes[i:i + batch_size]
				model = model.partial_fit(data, out.A1)

			y_pred = model.predict(X[2])
			avg_test_errs = np.append(avg_test_errs, [
				regression_performance(y[2].A1, y_pred, 'rms')
				])
		err = avg_test_errs.mean()
		if min_test_err is None or err < min_test_err:
			min_test_err = err
			best_epsilon = e

	print 'Enter range of epochs.'
	start = raw_input('\tEnter lower bound (inclusive) of range: ')
	end = raw_input('\tEnter upper bound (exclusive) of range: ')
	incr = raw_input('\tEnter increment: ')
	epoch_range = np.arange(int(start), int(end), int(incr))
	min_test_err = None
	best_epochs = None
	for e in epoch_range:
		avg_test_errs = np.array([])
		for seed in seeds[1:]:
			model = PassiveAggressiveRegressor(
				random_state=np.random.RandomState(seed)
				)
			for rnd in xrange(e):
				for i in xrange(0, total_training_instances, batch_size):
					data = train_data[i:i + batch_size]
					out = outcomes[i:i + batch_size]
					model = model.partial_fit(data, out.A1)

			y_pred = model.predict(X[2])
			avg_test_errs = np.append(avg_test_errs, [
				regression_performance(y[2].A1, y_pred, 'rms')
				])
		err = avg_test_errs.mean()
		if min_test_err is None or err < min_test_err:
			min_test_err = err
			best_epochs = e

	attributes = [
		'Perfusion Parameter',
		'Model',
		'Patch Radius',
		'Batch Size',
		'Total Number of Examples Trained',
		'Trial',
		'Training RMSE',
		'Test RMSE',
		'Training R^2 Score',
		'Test R^2 Score',
		'C (Regularization)', # Aggressiveness
		'Epsilon',
		'Epochs',
		'Fit Intercept?',
		'Shuffle?',
		'Loss Function',
		'Warm Start?'
	]
	results = {}
	for a in attributes:
		results[a] = []
	trials = raw_input('Enter number of trials: ')
	trials = int(trials)

	indices = np.arange(0, total_training_instances)
	np.random.shuffle(indices)
	train_data = np.matrix([np.asarray(X[0])[i] for i in indices])
	outcomes = np.matrix([[y[0].A1[i]] for i in indices])

	np.random.seed(None)
	for trial in xrange(trials):
		model = PassiveAggressiveRegressor(
			C=best_C,
			epsilon=best_epsilon,
			fit_intercept=True,
			n_iter=1, # Not applicable for partial fit
			shuffle=True,
			random_state=None,
			loss='epsilon_insensitive',
			warm_start=False
			)
		for rnd in xrange(best_epochs):
			for i in xrange(0, total_training_instances, batch_size):
				data = train_data[i:i + batch_size]
				out = outcomes[i:i + batch_size]
				model = model.partial_fit(data, out.A1)

		results['Perfusion Parameter'].append(perfusion_param)
		results['Model'].append('PA')
		results['Patch Radius'].append(patch_radius)
		results['Batch Size'].append(batch_size)
		results['C (Regularization)'].append(best_C)
		results['Epsilon'].append(best_epsilon)
		results['Fit Intercept?'].append(True)
		results['Shuffle?'].append(True)
		results['Epochs'].append(best_epochs)
		results['Trial'].append(trial + 1)
		results['Loss Function'].append('epsilon_insensitive')
		results['Warm Start?'].append(False)
		results['Total Number of Examples Trained'].append(
			total_training_instances
			)

		y_pred = model.predict(X[0])
		results['Training RMSE'].append(
			regression_performance(y[0].A1, y_pred, 'rms')
			)
		results['Training R^2 Score'].append(
			regression_performance(y[0].A1, y_pred, 'r2-score')
			)

		y_pred = model.predict(X[2])
		results['Test RMSE'].append(
			regression_performance(y[2].A1, y_pred, 'rms')
			)
		results['Test R^2 Score'].append(
			regression_performance(y[2].A1, y_pred, 'r2-score')
			)
	record_results(results, attributes, **{
		'title': 'trial results'
		})

def run_PA(X, y, **kwargs):
	"""
	Runs the Passive-Aggressive algorithm manually on the regression data.

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
	controls = ctrls()

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
		'Training RMSE',
		'Test RMSE',
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
	size = raw_input('Enter batch size (default: ' + str(batch_size) + '): ')
	if size == controls['Quit']:
		return
	if size != '':
		batch_size = int(size)

	# Enter a seed in order to reproduce results (even if the shuffle option is
	# not set to True)
	seed = None # Default
	pick_seed = raw_input('Enter seed for random number generator: ')
	if pick_seed == controls['Quit']:
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
			start = raw_input('\tEnter lower bound (inclusive) of range: ')
			end = raw_input('\tEnter upper bound (exclusive) of range: ')
			incr = raw_input('\tEnter increment: ')
			C_range = np.arange(float(start), float(end), float(incr))

			avg_train_perf = []
			avg_val_perf = []

			for c in C_range:
				train_perf = np.array([])
				val_perf = np.array([])

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
					train_perf = np.append(train_perf, [
						regression_performance(
						y_train,
						y_pred,
						'rms'
						)])
					y_pred = model.predict(X_val)
					val_perf = np.append(val_perf, [
						regression_performance(
						y_val,
						y_pred,
						'rms'
						)])
				avg_train_perf.append(train_perf.mean())
				avg_val_perf.append(val_perf.mean())
			plot_hyperparameter(
				C_range,
				avg_train_perf,
				avg_val_perf,
				**{
					'parameter' : r'Regularization $C$',
					'score' : 'Root Mean Squared Error'
				})
		elif comp == controls['Quit']:
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
			start = raw_input('\tEnter lower bound (inclusive) of range: ')
			end = raw_input('\tEnter upper bound (exclusive) of range: ')
			incr = raw_input('\tEnter increment: ')
			epsilon_range = np.arange(float(start), float(end), float(incr))
			avg_train_perf = []
			avg_val_perf = []

			for e in epsilon_range:
				train_perf = np.array([])
				val_perf = np.array([])

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
					train_perf = np.append(train_perf, [
						regression_performance(
						y_train,
						y_pred,
						'rms'
						)])
					y_pred = model.predict(X_val)
					val_perf = np.append(val_perf, [
						regression_performance(
						y_val,
						y_pred,
						'rms'
						)])
				avg_train_perf.append(train_perf.mean())
				avg_val_perf.append(val_perf.mean())
			plot_hyperparameter(
				epsilon_range,
				avg_train_perf,
				avg_val_perf,
				**{
					'parameter' : r'Epsilon $\epsilon$',
					'score' : 'Root Mean Squared Error'
				})
		elif comp == controls['Quit']:
			return
		else:
			break

	best_epsilon = 0.1 # Default
	epsilon_pick = raw_input('Choose value of epsilon (default: ' +
							 str(best_epsilon) + '): ')
	if epsilon_pick == controls['Quit']:
		return
	if epsilon_pick != '':
		best_epsilon = float(epsilon_pick)


	# shuffle
	shuffle = True # Default
	shuffle_pick = raw_input('Shuffle after each epoch (default: Y)? [Y/n] ')
	if shuffle_pick == controls['Quit']:
		return
	if shuffle_pick == 'n':
		shuffle = False


	# fit_intercept
	intercept = True # Default
	pick_intercept = raw_input('Fit intercept (default: Y)? [Y/n] ')
	if pick_intercept == controls['Quit']:
		return
	if pick_intercept == 'n':
		intercept = False


	# loss: 'epsilon_insensitive', 'squared_epsilon_insensitive'
	loss = 'epsilon_insensitive' # Default
	print 'Choose a loss function to be used.'
	print '1: Epsilon Insensitive (PA-I)'
	print '2: Squared Epsilon Insensitive (PA-II)'
	pick_loss = raw_input('Enter value (default: 1): ')
	if pick_loss == controls['Quit']:
		return
	if pick_loss == '2':
		loss = 'squared_epsilon_insensitive'


	##########################################################################
	# Observe performance of model for each batch that has been trained on.
	##########################################################################
	resp = raw_input('See incremental performance? [Y/n] ')
	if resp == controls['Quit']:
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
				'rms'
				)
			results['Training RMSE'].append(overall_train_perf)
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
				'rms'
				)
			results['Test RMSE'].append(test_perf)
			test_perf = regression_performance(
				y[2].A1,
				y_pred,
				'r2-score'
				)
			results['Test R^2 Score'].append(test_perf)

		learning_curve(
			incremental_sizes,
			results['Training RMSE'],
			results['Test RMSE'],
			**{ 'score' : 'Root Mean Squared Error' }
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
		print
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
		print 'Number of Epochs                      : 1'
		print ('Final Training Root Mean Squared Error: ' +
				str(final_result['Training RMSE'][0]))
		print ('Final Training R^2 Score              : ' +
				str(final_result['Training R^2 Score'][0]))
		print ('Final Test Root Mean Squared Error    : ' +
				str(final_result['Test RMSE'][0]))
		print ('Final Test R^2 Score                  : ' +
				str(final_result['Test R^2 Score'][0]))
		print

		record_results(results, attributes, **{
			'title': 'incremental results'
			})

	##########################################################################
	# Observe performance of model with more than 1 epoch.
	##########################################################################
	
	resp = raw_input('See performance with more than one epoch?\n'
					 'Parameters will remain the same. [Y/n] ')
	if resp == controls['Quit']:
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
				start = raw_input('\tEnter lower bound (inclusive) of range: ')
				end = raw_input('\tEnter upper bound (exclusive) of range: ')
				incr = raw_input('\tEnter increment: ')
				n_range = np.arange(int(start), int(end), int(incr))
				avg_train_perf = []
				avg_val_perf = []

				np.random.seed(seed)
				indices_copy = copy.deepcopy(indices)
				for n in n_range:
					train_perf = np.array([])
					val_perf = np.array([])

					if shuffle:
						np.random.shuffle(indices_copy)
						train_data = np.matrix(
							[np.asarray(X[0])[i] for i in indices_copy]
							)
						outcomes = np.matrix(
							[[y[0].A1[i]] for i in indices_copy]
							)

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
						train_perf = np.append(train_perf, [
							regression_performance(
							y_train,
							y_pred,
							'rms'
							)])
						y_pred = model.predict(X_val)
						val_perf = np.append(val_perf, [
							regression_performance(
							y_val,
							y_pred,
							'rms'
							)])
					avg_train_perf.append(train_perf.mean())
					avg_val_perf.append(val_perf.mean())
				plot_hyperparameter(
					n_range,
					avg_train_perf,
					avg_val_perf,
					**{
						'parameter' : r'Epochs',
						'score' : 'Root Mean Squared Error'
					})
			elif comp == controls['Quit']:
				return
			else:
				break

		best_n_iter = 5 # Default
		n_iter = raw_input('Enter the number of epochs (default: ' +
							str(best_n_iter) + '): ')
		if n_iter == controls['Quit']:
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
			'rms'
			)
		final_result['Training RMSE'].append(overall_train_perf)
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
			'rms'
			)
		final_result['Test RMSE'].append(test_perf)
		test_perf = regression_performance(
			y[2].A1,
			y_pred,
			'r2-score'
			)
		final_result['Test R^2 Score'].append(test_perf)
		final_result['Epochs'] = [best_n_iter]

		if 'Epochs' not in attributes:
			attributes.append('Epochs')

		# Print summary
		print
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
		print 'Number of Epochs                      : ' + str(best_n_iter)
		print ('Final Training Root Mean Squared Error: ' +
				str(final_result['Training RMSE'][0]))
		print ('Final Training R^2 Score              : ' +
				str(final_result['Training R^2 Score'][0]))
		print ('Final Test Root Mean Squared Error    : ' +
				str(final_result['Test RMSE'][0]))
		print ('Final Test R^2 Score                  : ' +
				str(final_result['Test R^2 Score'][0]))
		print
		record_results(final_result, attributes, **{
			'title': 'final results'
			})

def home_PA(X, y, **kwargs):
	"""
	Requests an action from the user to run the Passive-Aggressive algorithm on
	the regression data.

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

	controls = ctrls()
	suc = True
	while True:
		if suc:
			print '#############'
			print '## PA Home ##'
			print '#############'
		print 'How would you like to run Passive-Aggressive (PA)?'
		print '1. Run manually.'
		print '2. Run multiple times to get an average performance.'
		print 'Quit to run on new data.'
		pa_op = raw_input('Enter value: ')
		if pa_op == controls['Help']:
			print_controls()
			suc = False
		elif pa_op == controls['Quit']:
			return
		elif pa_op == controls['Skip']:
			print ('Unable to skip. Press ' + controls['Quit'] +
				   ' to exit.')
			suc = False
		elif pa_op == controls['Home']:
			suc = False
		else:
			try:
				pa_op = int(pa_op)
				if pa_op not in [1, 2]:
					print 'Invalid value. Try again.'
				else:
					if pa_op == 1:
						run_PA(X, y, **kwargs)
						suc = True
					elif pa_op == 2:
						run_multiple_PA(X, y, **kwargs)
						suc = True
			except ValueError:
				suc = False
				print 'Invalid value. Try again.'
		print
