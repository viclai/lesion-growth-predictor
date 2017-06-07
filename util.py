from collections import defaultdict
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import pandas as pd
import os
import sys

# Functions to visualize data, plot graphs, and evaluate models go here.

def scatter_matrix(X, **kwargs):
	"""
	Plots a scatter matrix of the data.

	Parameters
	--------------------
		X -- numpy matrix of shape (n,d), features
	"""

	indexes = ['Example ' + str(i + 1) for i in xrange(X.shape[0])]
	if 'features' not in kwargs:
		features = ['Feature ' + str(i + 1) for i in xrange(X.shape[1])]
	else:
		features = kwargs.pop('features')
	d = {}
	X_arr = np.asarray(X.T)
	for feat, i in zip(X_arr, xrange(len(X_arr))):
		d[features[i]] = feat
	df = pd.DataFrame(d, index=indexes)
	pd.scatter_matrix(df)
	plt.show()

def pca_plot(X, y):
	"""
	Perform dimensionality reduction to visualize multi-dimensional input.

	Parameters
	--------------------
		X -- numpy matrix of shape (n,d), features
		y -- numpy matrix of shape (n,1), targets
	"""

	pass # TODO

def two_dimensional_slices(X, y, **kwargs):
	"""
	Plots 2-D slices of the data, with a specific attribute on the horizontal
	axis and the target on the vertical axis.

	Parameters
	--------------------
		X -- numpy matrix of shape (n,d), features
		y -- numpy matrix of shape (n,1), targets
	"""

	if 'color' not in kwargs:
		kwargs['color'] = 'b'

	instances = X.shape[1]
	if 'parameter_name' not in kwargs:
		y_label = 'Label Value'
	else:
		y_label = kwargs.pop('parameter_name') + ' Value'

	if 'x-label' not in kwargs:
		x_label = 'Feature Value'
	else:
		x_label = kwargs.pop('x-label')

	plt.ion()
	for i in xrange(instances):
		plt.scatter(X[:,i].A1, y.A1, **kwargs)
		plt.xlabel(x_label, fontsize=16)
		plt.ylabel(y_label, fontsize=16)
		plt.draw()
		plt.pause(0.001)
		cont = raw_input("Press [C] to see next scatter plot. ")
		if cont != "C" and cont != "c":
			plt.close()
			break
		plt.clf()

def label_distribution(y, binsize=1, **kwargs):
	"""
	Plots a histogram displaying the distribution of the targets in the data.

	Parameters
	--------------------
		y       -- numpy matrix of shape (n,1), targets
		binsize -- size of the bin
	"""

	if 'color' not in kwargs:
		kwargs['color'] = 'b'

	if 'label_name' not in kwargs:
		label_name = 'Label Value'
	else:
		label_name = kwargs.pop('label_name') + ' Value'

	if 'title' not in kwargs:
		title = ''
	else:
		title = kwargs.pop('title')

	slots = [((slot / binsize) * binsize) + binsize if slot % binsize != 0
		else slot for slot in np.ceil(y).A1.astype(int)
		]
	freqs = defaultdict(int)
	for slot in slots:
		freqs[slot] += 1
	label_values = freqs.keys()
	frequencies = freqs.values()

	plt.clf()
	plt.bar(label_values, frequencies, **kwargs)
	plt.xlabel(label_name, fontsize=16)
	plt.ylabel('Frequency', fontsize=16)
	plt.title(title, fontsize=16)
	plt.ion()
	plt.draw()
	plt.pause(0.001)

def statistics(X, y, filename='stats.csv', **kwargs):
	"""
	Displays statistics relating to the data.

	Parameters
	--------------------
		X        -- numpy matrix of shape (n,d), features
		y        -- numpy matrix of shape (n,1), targets
		filename -- string, name of file to write to
	"""

	DISPLAY_MAX_ROWS = X.shape[0]
	pd.set_option("display.max_rows", DISPLAY_MAX_ROWS)
	if 'features' not in kwargs:
		digits = str(len(str(X.shape[1])))
		zeros_format = '0' + digits + 'd'
		features = ['Feature ' + format(i + 1, zeros_format) for i in
			xrange(X.shape[1])
		]
	else:
		features = kwargs.pop('features')
	d = {}
	X_arr = np.asarray(X.T)
	for feat, i in zip(X_arr, xrange(len(X_arr))):
		d[features[i]] = feat
	df = pd.DataFrame(d)

	dir = 'stats'
	if not os.path.exists(dir):
		os.makedirs(dir)
	file_path = os.path.join(dir, filename)

	avgs = df.apply(np.mean).values
	stds = df.apply(np.std).values
	d = { 'Mean' : avgs, 'Standard Deviation' : stds }
	df = pd.DataFrame(d, index=features)

	print ('Writing mean and standard deviation of each feature to ' +
		   file_path + '...'),
	df.to_csv(file_path, mode='w+')
	print 'Done.'

def regression_performance(y_true, y_pred, metric):
	"""
	Calculates the performance metric based on the agreement between the 
	true labels and the predicted labels.
    
	Parameters
	--------------------
		y_true -- numpy array of shape (n,), known labels
		y_pred -- numpy array of shape (n,), (continuous-valued) predictions
		metric -- string, option used to select the performance measure
				  options: 'mse', 'f1-mae', 'exp-var-score', 'r2-score', 'rms'
    
	Returns
	--------------------
		score  -- float, performance score
	"""

	if metric == "mse":
		return metrics.mean_squared_error(y_true, y_pred)
	elif metric == "mae":
		return metrics.mean_absolute_error(y_true, y_pred)
	elif metric == "exp-var-score":
		return metrics.explained_variance_score(y_true, y_pred)
	elif metric == "r2-score": # Can also use score() from the regression model
		return metrics.r2_score(y_true, y_pred)
	elif metric == "rms":
		return np.sqrt([metrics.mean_squared_error(y_true, y_pred)])[0]
	else:
		return 0

def plot_hyperparameter(h, train_sc, test_sc, **kwargs):
	"""
	Plots a scatterplot of the training score and test score versus a
	hyperparameter. This is helpful for hyperparameter tuning.

	Parameters
    --------------------
		h        -- list of length n, values of the hyperparameter
		train_sc -- list of length n, values of training score
		test_sc  -- list of length n, values of test score
	"""

	if 'parameter' not in kwargs:
		xlabel = 'Hyperparameter'
	else:
		xlabel = kwargs.pop('parameter')

	if 'score' not in kwargs:
		ylabel = 'Performance Score'
	else:
		ylabel = kwargs.pop('score')

	plt.clf()
	plt.scatter(h, train_sc, color='b', marker='x', label='Training')
	plt.scatter(h, test_sc, color='r', label='Test')
	plt.legend(loc='upper right', numpoints=1, title='Data Type')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(ylabel + ' of Training and Test Vs. ' + xlabel)
	plt.ion()
	plt.draw()
	plt.pause(0.001)

def learning_curve(size, train_sc, test_sc, **kwargs):
	"""
	Plots a scatterplot of the training score and test score versus the size
	of the data trained on. This is helpful for observing the performance of
	incremental/online models as more data is trained on.

	Parameters
    --------------------
		size     -- list of length n, values of increasing sizes
		train_sc -- list of length n, values of training score
		test_sc  -- list of length n, values of test score
	"""

	if 'score' not in kwargs:
		ylabel = 'Performance Score'
	else:
		ylabel = kwargs.pop('score')
	xlabel = 'Number of Training Examples'

	plt.clf()
	plt.scatter(size, train_sc, color='b', marker='x', label='Training')
	plt.scatter(size, test_sc, color='r', label='Test')
	plt.legend(loc='upper right', numpoints=1, title='Data Type')
	plt.xlabel(xlabel)
	plt.ylabel(ylabel)
	plt.title(ylabel + ' of Training and Test Vs. ' + xlabel)
	plt.ion()
	plt.draw()
	plt.pause(0.001)

def record_results(res, order=None, dir='results', **kwargs):
	"""
	Records results to a CSV file.

	Parameters
    --------------------
		res   -- dictionary of column names mapped to a list of their
        		 respective values
		order -- list of strings, values to order the columns by
		dir   -- string, name of directory to put results in
	"""

	if 'title' not in kwargs:
		name = 'results'
	else:
		name = kwargs.pop('title')

	df = pd.DataFrame(res)
	if order is not None:
		df = df[order]

	if not os.path.exists(dir):
		os.makedirs(dir)

	option_str = [
		'Do not record',
		'Write to new file',
		'Write to existing file',
	]

	files = [
		f for f in os.listdir(dir) if os.path.isfile(os.path.join(dir, f))
	]
	if len(files) == 0:
		print 'There are no existing files in ' + dir + '.'
		options = [0, 1]
	else:
		print 'Here is a list of files in ' + dir + ': '
		for f in files:
			print '\t' + f
		options = [0, 1, 2]

	while True:
		print 'How should the ' + name + ' be recorded?'
		for o in options:
			print str(o) + ': ' + option_str[o]
		resp = raw_input('Enter option number: ')
		try:
			val = int(resp)
			if val not in options:
				print ('Invalid value. Enter the value corresponding to the '
					   'option.')
			else:
				break
		except:
			print ('Invalid value. Enter the value corresponding to the '
				   'option.')

	if val == 1:
		filename = raw_input('Enter new file name: ')
	elif val == 2:
		while True:
			filename = raw_input('Enter existing file name: ')
			file_path = os.path.join(dir, filename)
			if not os.path.exists(file_path):
				new_file = raw_input('File does not exist. '
									 'Create new file? [Y/n] ')
				if new_file == 'Y':
					break
			else:
				break
	else:
		print name.title() + ' not recorded.'
		return

	file_path = os.path.join(dir, filename)

	include_header = False
	if not os.path.exists(file_path):
		include_header = True
	df.to_csv(
		file_path,
		header=include_header,
		mode='a',
		na_rep='N/A',
		index=False
		)
	print name.title() + ' written to ' + file_path + '.'

def scatter_plot_from_csv(filepath, split, attr, **kwargs):
	"""
	Displays a scatter plot from a CSV file.

	Parameters
	--------------------
		filepath -- path to CSV
		split    -- string, attribute (column) to split on
		attr     -- list of strings, values of attributes to plot
	"""

	"""
	Add more markers and/or colors if needed.
	Google 'matplotlib markers' and 'matplotlib colors'.
	"""
	markers = ['x', 'o']
	colors = ['b', 'r', 'g']

	if 'title' not in kwargs:
		title = ''
	else:
		title = kwargs.pop('title')

	if 'ylabel' not in kwargs:
		ylabel = 'y'
	else:
		ylabel = kwargs.pop('ylabel')

	try:
		df = pd.read_csv(filepath)
	except IOError:
		sys.stderr.write('Unknown file path\n')
		sys.exit()

	categories = df[split].unique()
	one_class = len(categories) == 1
	patches = []
	lines = []

	plt.clf()
	ax = plt.subplot(111)
	for i, category in enumerate(categories):
		patches.append(mpatches.Patch(color=colors[i], label=category))
		cur_df = df.loc[df[split] == category]
		d = cur_df.to_dict('list')

		x = d[attr[0]]
		for j, a in enumerate(attr[1:]):
			if one_class:
				color = colors[j]
			else:
				color = colors[i]
			ax.scatter(x, d[a], color=color, marker=markers[j], label=a)

			if i == 0:
				lines.append(mlines.Line2D(
					[],
					[],
					color='k',
					marker=markers[j],
					label=a
					))

	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

	if not one_class:
		class_legend = ax.legend(
			loc='lower left',
			fontsize=8,
			numpoints=1,
			handles=patches,
			bbox_to_anchor=(1, 0.25),
			title=split
			)
		class_legend.get_title().set_fontsize('8')
		plt.gca().add_artist(class_legend)
	ax.legend(
		loc='center left',
		fontsize=8,
		numpoints=1,
		handles=lines,
		bbox_to_anchor=(1, 0.75)
		)
	plt.xlabel(attr[0], **kwargs)
	plt.ylabel(ylabel, **kwargs)
	plt.title(title, **kwargs)
	plt.ion()
	plt.draw()
	plt.pause(0.001)

def histogram_from_csv(filepath, split, xlabel, ylabel, **kwargs):
	"""
	Displays a histogram (bar graph) from a CSV file.

	Parameters
	--------------------
		filepath -- path to CSV
		split    -- string, attribute (column) to split on
		xlabel   -- string, label of horizontal axis
		ylabel   -- string, label of vertical axis
	"""

	# Add more colors if needed. Google 'matplotlib colors'.
	colors = ['b', 'r', 'g', 'c', 'm', 'y', 'k']

	if 'title' not in kwargs:
		title = ''
	else:
		title = kwargs.pop('title')

	try:
		df = pd.read_csv(filepath)
	except IOError:
		sys.stderr.write('Unknown file path\n')
		sys.exit()

	categories = df[split].unique()
	labels = df[xlabel].unique()
	patches = []
	space = 2
	x = np.arange(0, space * len(labels), space)
	bin_width = .2
	total_width = bin_width * len(categories)

	plt.clf()
	plt.subplot(111)
	for i, category in enumerate(categories):
		patches.append(mpatches.Patch(color=colors[i], label=category))
		cur_df = df.loc[df[split] == category]
		d = cur_df.to_dict('list')

		y = []
		for label in labels:
			cc_df = cur_df.loc[cur_df[xlabel] == label]
			if len(cc_df) != 1:
				print 'More than 1 value'
				return
			d = cc_df.to_dict('list')
			y.append(d[ylabel][0])
		plt.bar(x + (bin_width * i), y, color=colors[i], width=bin_width)

	legend = plt.legend(loc='best', fontsize=8, handles=patches, title=split)
	legend.get_title().set_fontsize('8')
	plt.xticks(x + ((bin_width / 2) * (len(categories) - 1)), labels)
	plt.xlabel(xlabel, **kwargs)
	plt.ylabel(ylabel, **kwargs)
	plt.title(title, **kwargs)
	plt.ion()
	plt.draw()
	plt.pause(0.001)
