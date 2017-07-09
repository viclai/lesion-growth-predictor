from enum import Enum
from collections import OrderedDict
from extracted_data.uniform_sampled.ten_k_subset \
    import matrix_loader as ml
from util import two_dimensional_slices, label_distribution, \
                 scatter_matrix, statistics
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import sys
import os

class NoDataError(Exception):
    def __init__(self, msg='No data loaded'):
        self.msg = msg


class DataSet(object):
    class DataType(Enum):
        TRAIN = 0
        VALIDATION = 1
        TEST = 2

    def __init__(self, X=(None, None, None), y=(None, None, None)):
        """
        Data class.
        
        Attributes
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

        # n = number of examples, d = dimensionality
        self.X = X
        self.y = y

    def load(self):
        raise NotImplementedError()

    def plot(self):
        raise NotImplementedError()


class PerfusionDataSet(DataSet):
    perfusion_params = list(ml.Matrix_loader().mets.keys())
    patch_radiuses = [int(r[0]) for r in ml.Matrix_loader().halfwindowsizes]
    path_to_dataset = 'extracted_data/uniform_sampled/ten_k_subset'
    uniform_bins = {
        'rbf'  : 2,
        'rbv'  : 1,
        'mtt'  : 1,
        'ttp'  : 1,
        'tmax' : 1
    }
    # Number of seconds that separates each concentration level
    time_interval = 2

    def __init__(self, perfusion_param=None, patch_radius=None, dt=None):
        """
        PerfusionData class.

        Attributes
        --------------------
            perfusion_param -- string, perfusion parameter
            patch_radius    -- integer, patch radius
            dt              -- DataType, type of dataset to load data
                               to
        """

        DataSet.__init__(self)
        self.perfusion_param = perfusion_param
        self.patch_radius = patch_radius
        self.type = dt

    def load(self, perfusion_param, patch_radius, dt):
        """
        Load memory-mapped files into X array of features and y array
        of labels.

        Parameters
        --------------------
            perfusion_param -- string, perfusion parameter
            patch_radius    -- integer, radius of patch size
            dt              -- DataType, type of dataset to load data
                               to
        """

        m = ml.Matrix_loader(self.path_to_dataset)
        train_X, train_y, val_X, val_y = m.load(
            perfusion_param,
            patch_radius
            )

        X = list(self.X)
        y = list(self.y)

        if dt == self.DataType.TRAIN:
            X[dt.value] = train_X
            self.X = tuple(X)
            y[dt.value] = train_y
            self.y = tuple(y)
        elif dt == self.DataType.VALIDATION:
            X[dt.value] = val_X
            self.X = tuple(X)
            y[dt.value] = val_y
            self.y = tuple(y)
        elif dt == self.DataType.TEST:
            X[dt.value] = val_X
            self.X = tuple(X)
            y[dt.value] = val_y
            self.y = tuple(y)

        self.perfusion_param = perfusion_param
        self.patch_radius = patch_radius
        self.type = dt

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
        time = np.arange(0, points * self.time_interval, self.time_interval)
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

    def tissue_curve(self, dt, **kwargs):
        """
        Plots the CTC for each instance.

        Parameters
        --------------------
            dt -- DataType, type of dataset to plot
        """

        X = self.X[dt.value]

        # All columns excluding last 40 columns of entire data set
        instances = X.shape[1] - 40
        self.concentration_time_curve(X[:,:instances], **kwargs)

    def aterial_input_function(self, dt, **kwargs):
        """
        Plots the CTC of the AIF for each instance.

        Parameters
        --------------------
            dt -- DataType, type of dataset to plot
        """

        X = self.X[dt.value]
        instances = 40 # Last 40 columns of entire data set
        self.concentration_time_curve(X[:,-instances:], **kwargs)

    def plot(self, type, dt, **kwargs):
        """
        Plots a specified graph to visualize the data.

        Parameters
        --------------------
            type     -- string, type of graph/histogram to plot
                        options: 'ctc', 'tissue-curve', 'aif', 
                        '2d-slices', 'label_distribution',
                        'scatter_matrix', 'stats'
            dt       -- DataType, type of dataset to plot
        """

        X = self.X[dt.value]
        y = self.y[dt.value]

        if X is None or y is None:
            raise NoDataError()

        type_str = {
            0 : 'Training',
            1 : 'Validation',
            2 : 'Test'
        }[dt.value]

        if type == 'ctc':
            # Plot tissue curve with AIF curve
            self.concentration_time_curve(X, **kwargs)
        elif type == 'tissue_curve':
            self.tissue_curve(dt, **kwargs)
        elif type == 'aif':
            self.aterial_input_function(dt, **kwargs)
        elif type == '2d-slices':
            kwargs['parameter_name'] = self.perfusion_param
            kwargs['x-label'] = 'Concentration Level'
            two_dimensional_slices(X, y, **kwargs)
        elif type == 'label_distribution':
            kwargs['label_name'] = self.perfusion_param
            kwargs['title'] = ('Frequency Distribution of ' +
                self.perfusion_param + ' (' + type_str + ')')
            label_distribution(y, self.uniform_bins[self.perfusion_param],
                **kwargs)
        elif type == 'scatter_matrix':
            kwargs['features'] = ['Time ' + str(i) for i in 
                xrange(0, self.time_interval * X.shape[1], self.time_interval)
                ]
            scatter_matrix(X, **kwargs)
        elif type == 'stats':
            digits = str(len(str(self.time_interval * X.shape[1])))
            zeros_format = '0' + digits + 'd'
            kwargs['features'] = ['Time ' + format(i, zeros_format) for i in
                xrange(0, self.time_interval * X.shape[1], self.time_interval)
            ]
            filename = ('stats-' + self.perfusion_param + '_' + 
                        str(self.patch_radius) + '_' + type_str + '.csv')
            statistics(X, y, filename=filename, **kwargs)
        else:
            print 'No such plot exists\n'

    def write_amazon_ml_schema(self, dir='amazon_ml'):
        """
        Generates a data schema for Amazon ML.

        Parameters
        --------------------
            dir -- string, the directory to save the schema to
        """

        if self.perfusion_param is None:
            raise TypeError('Perfusion parameter not specified')
        if self.patch_radius is None:
            raise TypeError('Patch radius not specified')

        schema = OrderedDict()
        schema['version'] = '1.0'
        schema['rowId'] = 'brainPixelId'
        schema['targetAttributeName'] = self.perfusion_param
        schema['dataFormat'] = 'CSV'
        schema['dataFileContainsHeader'] = False

        schema['attributes'] = []
        schema['attributes'].append(
            dict(attributeName='brainPixelId', attributeType='CATEGORICAL')
            )
        for i in range((1 + (self.patch_radius * 2)) ** 2):
            for j in range(40):
                attr = {}
                attr['attributeName'] = ('CTC ' + str(i + 1) + ' - Time ' +
                    str(j * self.time_interval))
                attr['attributeType'] = 'NUMERIC'
                schema['attributes'].append(attr)

        for i in range(40):
            attr = {}
            attr['attributeName'] = ('AIF - Time ' +
                str(i * self.time_interval))
            attr['attributeType'] = 'NUMERIC'
            schema['attributes'].append(attr)
        schema['attributes'].append(
            dict(attributeName=self.perfusion_param, attributeType='NUMERIC')
            )

        filename = (self.perfusion_param + '_' + str(self.patch_radius) +
            '_data.csv.schema')
        if not os.path.exists(dir):
            os.makedirs(dir)
        file_path = os.path.join(dir, filename)
        with open(file_path, 'w') as outfile:
            json.dump(schema, outfile, indent=4, separators=(',', ':'))

    def write_to_csv(self, dir='csv'):
        """
        Saves the data in a CSV file.
        This can be used as an Amazon ML datasource, which can hold
        rows of observations each limited to size of 100 KB.

        Parameters
        --------------------
            dir -- string, the directory to save the CSV file to
        """

        df = pd.DataFrame(self.X[self.type.value])
        df['y'] = self.y[self.type.value]

        filename = (self.perfusion_param + '_' + str(self.patch_radius) +
            '_data.csv')
        if not os.path.exists(dir):
            os.makedirs(dir)
        file_path = os.path.join(dir, filename)
        print 'Writing data to ' + file_path + '...',
        sys.stdout.flush()
        df.to_csv(file_path, header=False)
        print 'Done.'
