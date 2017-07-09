from data import PerfusionDataSet, NoDataError
from models import home_SGD, home_PA
from controls import print_controls, ctrls
import sys
import numpy as np

home_options = [
    'Evaluate perfusion parameter.',
    'Plot from existing results.',
    'Exit program.'
]

def evaluate_param():
    controls = ctrls()
    home = True
    while True:
        if home:
            print '#################################'
            print '## Parameter Evaluation - Home ##'
            print '#################################'
        home = False

        ###############################################################
        # Load data
        ###############################################################
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
                return
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
        patch_rad_range = p_data.patch_radiuses
        while True:
            patch_rad = raw_input('Enter patch radius (0-6): ')
            if patch_rad == controls['Skip']:
                print 'Patch radius is required.'
            elif patch_rad == controls['Quit']:
                return
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

        print 'Loading data...',
        sys.stdout.flush()
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
        print 'Done'

        if home == True:
            continue
        print

        ###############################################################
        # Visualize data
        ###############################################################
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
                return
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
                        return
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
                                return
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
                                            again = raw_input('Display ' + 
                                                name + ' for another type of '
                                                'data? [Y/n] ')
                                            if again == controls['Skip'] or \
                                                again == 'n':
                                                again = False
                                                break
                                            elif again == controls['Quit']:
                                                return
                                            elif again == controls['Help']:
                                                print_controls()
                                            elif again == controls['Home']:
                                                home = True
                                                break
                                            elif again == 'Y':
                                                again = True
                                                break
                                            else:
                                                print ('Invalid response. '
                                                      'Try again.')
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

        ###############################################################
        # Run models
        ###############################################################

        ### Stochastic Gradient Descent Regressor ###
        while True:
            exe = raw_input('Run Stochastic Gradient Descent? [Y/n] ')
            if exe == 'n' or exe == controls['Skip']:
                break
            elif exe == controls['Quit']:
                return
            elif exe == controls['Help']:
                print_controls()
            elif exe == controls['Home']:
                home = True
                break
            elif exe == 'Y':
                kwargs = {
                    'parameter' : param_name,
                    'patch_radius' : patch_rad
                }
                home_SGD(p_data.X, p_data.y, **kwargs)
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
                return
            elif exe == controls['Help']:
                print_controls()
            elif exe == controls['Home']:
                home = True
                break
            elif exe == 'Y':
                kwargs = {
                    'parameter' : param_name,
                    'patch_radius' : patch_rad
                }
                home_PA(p_data.X, p_data.y, **kwargs)
                break
            else:
                print 'Invalid response. Try again.'
        print

        while True:
            resp = raw_input('Run on another data set? [Y/n] ')
            if resp == controls['Quit'] or resp == 'n' or \
                 resp == controls['Skip']:
                return
            elif resp == controls['Help']:
                print_controls()
            elif resp == controls['Home']:
                home = True
                break
            elif resp == 'Y':
                break
            else:
                print 'Invalid response. Try again.'

def plot():
    print 'Plot not implemented yet. Returning home.'

def main():
    print 'Welcome to Lesion Growth Predictor!'
    print_controls()

    controls = ctrls()
    suc = True
    while True:
        if suc:
            print '#################'
            print '## MAIN - Home ##'
            print '#################'
        print 'What would you like to do?'
        for i, op in enumerate(home_options):
            print str(i + 1) + ': ' + op
        home_op = raw_input('Enter value: ')
        if home_op == controls['Skip']:
            print ('Unable to skip. Press ' + controls['Quit'] +
                   ' to exit program.')
            suc = False
        elif home_op == controls['Quit']:
            print 'Exiting program.'
            sys.exit()
        elif home_op == controls['Help']:
            print_controls()
            suc = False
        elif home_op == controls['Home']:
            suc = False
        else:
            try:
                home_op = int(home_op)
                if home_op not in xrange(1, len(home_options) + 1):
                    print 'Invalid value. Try again.'
                else:
                    print
                    if home_op == 1:
                        evaluate_param()
                        suc = True
                    elif home_op == 2:
                        plot()
                        suc = True
                    elif home_op == 3:
                        print 'Exiting program.'
                        sys.exit()
                    print
            except ValueError:
                suc = False
                print 'Invalid value.'

if __name__ == "__main__" :
    main()
