import numpy as np
import pandas as pd
from utils.subroutines import subroutine_familywise, subroutine
from utils.multiple_testing import eBH, e_function_static
from utils.helper_functions import compute_fdp, compute_power
from utils.generating_data import gen_data

from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.svm import OneClassSVM


#HYPERPARAMETERS

alpha_fdr = 0.05
m = 50 #number of points in test
n = 200 #number of points in the calibration
prop_outliers = 0.5 #proportion of outlier (pi) in the experiement
amp = 3.0 #The amplitude in the training data, refer to notes
K_blocks = 20
n_exp = 100

#Baseline model
KFC_classifier = OneClassSVM(nu=0.8, kernel='rbf', gamma='scale')  # initialize base model

#We construct the family of models
model_parameters = [0.1,0.3,0.5,0.7,0.9]
model_family = []

for nu in model_parameters:
    model_family.append(OneClassSVM(nu=nu, kernel='rbf', gamma='scale'))

#We set up important lists
settings = {'amp': 4.0, 'amps': [2.0, 2.5, 3.0, 3.5, 4.0], 'prop_outliers': 0.2, 'props_outliers': [0.1, 0.2, 0.3, 0.4] }
results_power = {'KFC':[], 'Family': []}
results_fdr = {'KFC':[], 'Family': []}

# details
amps = np.arange(2,4,0.2) # we create a range of parameters to visualize power curve
n_train = n # can specify training amount 
adjustment = 1/(1+alpha_fdr)

# The synthetic data
THETA = np.zeros(50) 

#The setting we're planning to investigate
p_theta = 6 
THETA[:p_theta,] = np.array([0.3, 0.3, 0.2, 0.2, 0.1, 0.1]) 
THETA = THETA.reshape((50,1))


# FDR estimation
adjustment = 0.5
amp = settings['amp']
for prop_outliers in settings['props_outliers']:
    fdp_KFC = []
    fdp_family = []
    for exp in range(n_exp):
        np.random.seed(exp)
        #We repeat the same structure of the experiement
        Wset = np.random.uniform(size=(50,50)) * 6 - 3
        n_outliers = int(np.ceil(m * prop_outliers))
        blocks = np.array_split(np.arange(m), K_blocks) 
        
        # generate null and alternative hypotheses
        Xref = gen_data(Wset, n, 1)
        perm = np.random.permutation(m) 
        nonzero = np.sort(perm[:n_outliers])
        nulls = np.sort(perm[n_outliers:])
        
        # don't generate test dataset under covariate shift
        Xtest0 = gen_data(Wset, m-n_outliers, 1)    # inliers 
        Xtest1 = gen_data(Wset, n_outliers, amp)    # outliers
        
        # weights are just ones vector
        test_weights = np.ones(m)
        cal_weights = np.ones(n)
        
        Xtest = np.zeros((m, Xtest0.shape[1]))
        Xtest[nonzero,] = Xtest1
        Xtest[nulls,] = Xtest0
        
        e_rej_family = subroutine_familywise(model_family, Xref, Xtest, blocks, alpha_fdr)
        baseline_score_matrix = subroutine(KFC_classifier, Xref, Xtest, blocks) 
        
        e_values_KFC = e_function_static(adjustment*alpha_fdr, baseline_score_matrix, cal_weights, test_weights)
        e_rej = eBH(e_values_KFC, alpha_fdr)

        fdp_KFC.append(compute_fdp(e_rej, nulls))
        fdp_family.append(compute_fdp(e_rej_family, nulls))
        print(f'{exp}/{n_exp} Done')
    print(f'FDR KFC: {np.mean(fdp_KFC)}, FDR Family: {np.mean(fdp_family)}')
    results_fdr['KFC'].append(fdp_KFC)
    results_fdr['Family'].append(fdp_family)


prop_outliers = settings['prop_outliers']
for amp in settings['amps']:
    power_KFC = []
    power_family = []
    for exp in range(n_exp):
        np.random.seed(exp)
        #We repeat the same structure of the experiement
        Wset = np.random.uniform(size=(50,50)) * 6 - 3
        n_outliers = int(np.ceil(m * prop_outliers))
        blocks = np.array_split(np.arange(m), K_blocks) 
        
        # generate null and alternative hypotheses
        Xref = gen_data(Wset, n, 1)
        perm = np.random.permutation(m) 
        nonzero = np.sort(perm[:n_outliers])
        nulls = np.sort(perm[n_outliers:])
        
        # don't generate test dataset under covariate shift
        Xtest0 = gen_data(Wset, m-n_outliers, 1)    # inliers 
        Xtest1 = gen_data(Wset, n_outliers, amp)    # outliers
        
        # weights are just ones vector
        test_weights = np.ones(m)
        cal_weights = np.ones(n)
        
        Xtest = np.zeros((m, Xtest0.shape[1]))
        Xtest[nonzero,] = Xtest1
        Xtest[nulls,] = Xtest0
        
        e_rej_family =subroutine_familywise(model_family, Xref, Xtest, blocks, alpha_fdr)
        
        baseline_score_matrix = subroutine(KFC_classifier, Xref, Xtest, blocks) 
        e_values_KFC = e_function_static(adjustment*alpha_fdr, baseline_score_matrix, cal_weights, test_weights)
        e_rej = eBH(e_values_KFC, alpha_fdr)

        power_KFC.append(compute_power(e_rej, nonzero))
        power_family.append(compute_power(e_rej_family, nonzero))
        print(f'{exp}/{n_exp} Done')
    print(f'Finished {amp}, Power KFC: {np.mean(power_KFC)}, Power Family: {np.mean(power_family)}')
    results_power['KFC'].append(power_KFC)
    results_power['Family'].append(power_family)

FDR_KFC = []
err_KFC = []

FDR_family = []
err_family = []

power_KFC = []
err_power_KFC = []

power_family = []
err_power_family = []

#Saving the information
for idx, pi in enumerate(settings['props_outliers']):
    FDR_KFC.append(np.mean(results_fdr['KFC'][idx]))
    FDR_family.append(np.mean(results_fdr['Family'][idx]))
    err_KFC.append(np.std(results_fdr['KFC'][idx]))
    err_family.append(np.std(results_fdr['Family'][idx]))
for idx, amp in enumerate(settings['amps']):
    err_power_KFC.append(np.std(results_power['KFC'][idx]))
    err_power_family.append(np.std(results_power['Family'][idx]))
    power_KFC.append(np.mean(results_power['KFC'][idx]))
    power_family.append(np.mean(results_power['Family'][idx]))

#Save the dataframes
df_fdr = pd.DataFrame({
    'props_outliers': settings['props_outliers'],
    'FDR_KFC': FDR_KFC,
    'FDR_family': FDR_family,
    'err_KFC': err_KFC,
    'err_family': err_family
})
df_power = pd.DataFrame({
    'amps': settings['amps'],
    'power_KFC': power_KFC,
    'power_family': power_family,
    'err_power_KFC': err_power_KFC,
    'err_power_family': err_power_family
})

df_combined = pd.concat([df_fdr, df_power], axis=1)
df_combined.to_csv('simulation_results/family_simulation_results.csv', index=False)
