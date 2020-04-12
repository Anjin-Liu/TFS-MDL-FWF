import numpy as np
import matplotlib.pyplot as plt
import freqopttest.util as util
import freqopttest.data as data
import freqopttest.kernel as kernel
import freqopttest.tst as tst
import freqopttest.glo as glo
import theano
from numpy import genfromtxt


dataImported = genfromtxt('p_q_1.csv', delimiter=',')

X = dataImported[0:999, 0:3].copy()
Y = dataImported[1000:1999, 0:3].copy()

tr = data.TSTData(X, Y)

#tr, te = tst_data.split_tr_te(tr_proportion=0.5, seed=2)

# J = number of test locations (features)
J = 10

# significance level of the test
alpha = 0.05

# These are options to the optimizer. Many have default values. 
# See the code for the full list of options.
op = {
    'n_test_locs': J, # number of test locations to optimize
    'max_iter': 200, # maximum number of gradient ascent iterations
    'locs_step_size': 1.0, # step size for the test locations (features)
    'gwidth_step_size': 0.1, # step size for the Gaussian width
    'tol_fun': 1e-4, # stop if the objective does not increase more than this.
    'seed': 8,  # random seed
}

# optimize on the training set
test_locs, gwidth, info = tst.MeanEmbeddingTest.optimize_locs_width(tr, alpha, **op)

counter = 0;
for i in range(2,150):

	dataImported = genfromtxt('p_q_'+str(i)+'.csv', delimiter=',')

	X = dataImported[0:999, 0:3].copy()
	Y = dataImported[1000:1999, 0:3].copy()

	te = data.TSTData(X, Y)

	met_opt = tst.MeanEmbeddingTest(test_locs, gwidth, alpha)
	test_result = met_opt.perform_test(te)
	
	if test_result['h0_rejected']:
		counter = counter + 1
		print 'detected: '+str(counter)
	else:
		print 'run: '+str(i)+' alpha: ' +str(test_result['alpha'])

print 'total ' + str(counter)













