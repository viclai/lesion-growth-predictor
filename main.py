from data import PerfusionData
from models import run_SGD, run_PA
import sys

controls = {
	'Help' : 'h',
	'Home' : 'home',
	'Skip' : '',
	'Quit' : 'q'
}

special_keys = {
	'' : 'ENTER'
}

def print_controls():
	print 'Controls:'
	for name, key in controls.iteritems():
		if key in special_keys:
			key = special_keys[key]
		print '\t' + name + ': [' + key + ']'
	print

def main():
	while True:
		home = False
		print 'Welcome to Lesion Growth Predictor!'
		print_controls()

		######################################################################
		# Load data
		######################################################################
		p_data = PerfusionData()

		### Enter perfusion parameter ###
		perfusion_params = p_data.perfusion_params
		param_val = None
		param_val_range = [i + 1 for i in xrange(len(perfusion_params) - 1)]
		while True:
			print 'Indicate which perfusion parameter to evaulate.'
			for i, p in zip(xrange(len(perfusion_params)), perfusion_params):
				if i == 0: # Skip parameter option 'all'
					continue
				print str(i) + ': ' + p
			param_val = raw_input('Enter value: ')
			if param_val == controls['Skip']:
				print 'Perfusion parameter is required.'
			elif param_val == controls['Quit']:
				print 'Exiting program.'
				sys.exit()
			elif param_val == controls['Help']:
				print_controls()
			elif param_val == controls['Home']:
				home = True
				break
			else:
				try:
					param_val = int(param_val)
					if param_val not in param_val_range:
						print ('Invalid value. Enter the value corresponding '
					           'to the parameter name.')
					else:
						break
				except:
					print ('Invalid value. Enter the value corresponding to '
					       'the parameter name.')

		if home == True:
			continue

		param_name = perfusion_params[param_val]

		### Enter patch radius ###
		patch_rad = None
		patch_rad_range = [i for i in xrange(7)]
		while True:
			patch_rad = raw_input('Indicate patch radius (0-6): ')
			if patch_rad == controls['Skip']:
				print 'Patch radius is required.'
			elif patch_rad == controls['Quit']:
				print 'Exiting program.'
				sys.exit()
			elif patch_rad == controls['Help']:
				print_controls()
			elif patch_rad == controls['Home']:
				home = True
				break
			else:
				try:
					patch_rad = int(patch_rad)
					if patch_rad not in patch_rad_range:
						print 'Invalid patch radius. Try again.'
					else:
						break
				except:
					print 'Invalid patch radius. Try again.'

		if home == True:
			continue

		p_data.load(perfusion_params[param_val], patch_rad) 

		######################################################################
		# Visualize data
		######################################################################
		plots = {
			'Tissue and AIF Curve'          : 'ctc',
			'Tissue Curve'                  : 'tissue_curve',
			'Arterial Input Function (AIF)' : 'aif',
			'2D Slices'                     : '2d-slices',
			'Distribution of ' + param_name : 'label_distribution',
			'Scatter Matrix'                : 'scatter_matrix',
			'Statistics'                    : 'stats'
		}
		while True:
			visualize = raw_input('Visualize data? [Y/n] ')
			if visualize == controls['Skip'] or visualize == 'n':
				break
			elif visualize == controls['Quit']:
				print 'Exiting program.'
				sys.exit()
			elif visualize == controls['Help']:
				print_controls()
			elif visualize == controls['Home']:
				home = True
				break
			elif visualize == 'Y':
				for name, f_str in plots.iteritems():
					display = raw_input('Display ' + name + '? [Y/n] ')
					if display == 'n' or display == controls['Skip']:
						continue
					elif display == controls['Quit']:
						print 'Exiting program.'
						sys.exit()
					elif display == controls['Help']:
						print_controls()
					elif display == controls['Home']:
						home = True
						break
					elif display == 'Y':
						p_data.plot(f_str)
					else:
						print 'Invalid response. Try again.'
				break
			else:
				print 'Invalid response. Try again.'

		if home == True:
			continue

		######################################################################
		# Run models
		######################################################################

		### Stochastic Gradient Descent Regressor ###
		while True:
			exe = raw_input('Run Stochastic Gradient Descent? [Y/n] ')
			if exe == 'n' or exe == controls['Skip']:
				break
			elif exe == controls['Quit']:
				print 'Exiting program.'
				sys.exit()
			elif exe == controls['Help']:
				print_controls()
			elif exe == controls['Home']:
				home = True
				break
			elif exe == 'Y':
				run_SGD()
				break
			else:
				print 'Invalid response. Try again.'

		if home == True:
			continue

		### Passive-Aggressive Regressor ###
		while True:
			exe = raw_input('Run Passive-Aggressive? [Y/n] ')
			if exe == 'n' or exe == controls['Skip']:
				break
			elif exe == controls['Quit']:
				print 'Exiting program.'
				sys.exit()
			elif exe == controls['Help']:
				print_controls()
			elif exe == controls['Home']:
				home = True
				break
			elif exe == 'Y':
				run_PA()
				break
			else:
				print 'Invalid response. Try again.'

		while True:
			resp = raw_input('Run on another data set? [Y/n] ')
			if resp == controls['Quit'] or resp == 'n' or \
				 resp == controls['Skip']:
				print 'Exiting program.'
				sys.exit()
			elif resp == controls['Help']:
				print_controls()
			elif resp == controls['Home']:
				home = True
				break
			elif resp == 'Y':
				continue
			else:
				print 'Invalid response. Try again.'

if __name__ == "__main__" :
	main()
