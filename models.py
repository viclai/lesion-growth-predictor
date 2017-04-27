from sklearn.linear_model import SGDRegressor, PassiveAggressiveRegressor
from data import PerfusionData
import numpy as np

"""
Steps to evaluate a ML technique:
	1. Get the data!
	2. Visualize the data through plots.
	3. Train the model.
	4. Evaluate the model.
		-- Use cross validation to tune hyperparameters.
		-- Look out for overfitting and underfitting.
		-- Repeat Step 3 to find the best model.
"""

def train_SGD():
	print 'Examining Stochastic Gradient Descent for Linear Regression...'
	print 'Done'

def train_PA():
	print 'Examining Passive-Aggressive for Regression...'
	print 'Done'

def main():
	print 'Visualizing data...'
	p_data = PerfusionData()
	perfusion_params = p_data.perfusion_params
	p_data.load('tmax', 6)
	p_data.plot('2d-slices')

	train_SGD()
	train_PA()

if __name__ == "__main__" :
	main()
