from data import PerfusionDataSet, NoDataError
from models import run_SGD, run_PA
import sys
import numpy as np

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
		p_data = PerfusionDataSet()

		### Enter perfusion parameter ###
		perfusion_params = p_data.perfusion_params
		param_val = None
		param_val_range = [i for i in xrange(len(perfusion_params))]
		while True:
			print 'Enter which perfusion parameter to evaluate.'
			for i, p in enumerate(perfusion_params):
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
			patch_rad = raw_input('Enter patch radius (0-6): ')
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

		p_data.load(
			perfusion_params[param_val],
			patch_rad,
			PerfusionDataSet.DataType.TRAIN,
			)
		p_data.load(
			perfusion_params[param_val],
			patch_rad,
			PerfusionDataSet.DataType.TEST,
			)

		if home == True:
			continue
		print

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
						dts = {
							1 : PerfusionDataSet.DataType.TRAIN,
							2 : PerfusionDataSet.DataType.VALIDATION,
							3 : PerfusionDataSet.DataType.TEST
						}
						while True:
							print 'Which data to display ' + name + ' for?'
							print '1: Training'
							print '2: Validation'
							print '3: Test'
							data_type = raw_input('Enter value: ')
							if data_type == controls['Quit']:
								print 'Exiting program.'
								sys.exit()
							elif data_type == controls['Help']:
								print_controls()
							elif data_type == controls['Home']:
								home = True
								break
							else:
								try:
									data_type = int(data_type)
									if data_type not in dts.keys():
										print ('Invalid value. Enter the value'
											   ' corresponding to the data '
											   'type.')
									else:
										p_data.plot(f_str, dts[data_type])

										while True:
											again = raw_input('Display ' + name + 
												' for another type of data? [Y/n] ')
											if again == controls['Skip'] or \
												again == 'n':
												again = False
												break
											elif again == controls['Quit']:
												print 'Exiting program.'
												sys.exit()
											elif again == controls['Help']:
												print_controls()
											elif again == controls['Home']:
												home = True
												break
											elif again == 'Y':
												again = True
												break
											else:
												print 'Invalid response. Try again.'
								except ValueError:
									print ('Invalid value. Enter the value '
										   'corresponding to the data type.')
									again = True
								except NoDataError:
									print ('No data loaded to display. '
										   'Try another type.')
									again = True

								if not again or home == True:
									break

						if home == True:
							break
					else:
						print 'Invalid response. Try again.'
				break
			else:
				print 'Invalid response. Try again.'

		if home == True:
			continue
		print

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
				run_SGD(p_data.X, p_data.y)
				break
			else:
				print 'Invalid response. Try again.'

		if home == True:
			continue
		print

		### Passive-Aggressive Regressor ###
		while True:
			exe = raw_input('Run Passive-Aggressive (PA)? [Y/n] ')
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
				kwargs = {}
				kwargs['parameter'] = param_name
				kwargs['patch_radius'] = patch_rad
				run_PA(p_data.X, p_data.y, **kwargs)
				break
			else:
				print 'Invalid response. Try again.'
		print

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
				break
			else:
				print 'Invalid response. Try again.'
		print

if __name__ == "__main__" :
	main()
