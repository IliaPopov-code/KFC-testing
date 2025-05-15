from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import IsolationForest
from utils.subroutines import compute_beta, subroutine_with_twoclass, subroutine
from utils.helper_functions import compute_fdp, compute_power
from utils.generating_data import gen_data
from utils.multiple_testing import e_function_static, eBH
import numpy as np
import pandas as pd

#HYPERPARAMETERS
results_power = {'KFC':[], 'Two-class': []}
results_fdr = {'KFC':[], 'Two-class': []}

amps = np.arange(2.5,6,0.5)
alpha_fdr = 0.05
prop_outliers = 0.4

K_blocks = 20
n_exp = 100
m = 100
n = 1000 

beta = compute_beta(n, m, K_blocks)
adjustment = 1/(1 + alpha_fdr)
power_list = []

for amp in amps:

    power_KFC = []
    power_two_class = []

    fdp_KFC = []
    fdp_two_class = []
    
    for exp in range(n_exp):
        np.random.seed(exp)

        #We repeat the same structure of the experiement
        Wset = np.random.uniform(size=(50,50)) * 6 - 3
        n_outliers = int(np.ceil(m * prop_outliers))

        # generate null and alternative hypotheses
        perm = np.random.permutation(m) 
        nonzero = np.sort(perm[:n_outliers])
        nulls = np.sort(perm[n_outliers:])

        # ---------------- UNWEIGHTED -----------------------

        Xref = gen_data(Wset, n, 1)

        # don't generate test dataset under covariate shift
        Xtest0 = gen_data(Wset, m-n_outliers, 1)    # inliers 
        Xtest1 = gen_data(Wset, n_outliers, amp)    # outliers

        # weights are just ones vector
        test_weights = np.ones(m)
        cal_weights = np.ones(n)

        Xtest = np.zeros((m, Xtest0.shape[1]))
        Xtest[nonzero,] = Xtest1
        Xtest[nulls,] = Xtest0

        # ---------------------------------------------------

        KFC_classifier = IsolationForest(max_samples=1.0, n_estimators=50)  # initialize base model
        two_class_classifier = LogisticRegression()

        # test blocks used in combination with reference set for training 
        blocks = np.array_split(np.arange(m), K_blocks)    # split the D_test into K_blocks blocks
        
        # score matrix making subroutine
        real_score_matrix = subroutine(KFC_classifier, Xref, Xtest, blocks)    # original
        two_class_score_matrix = subroutine_with_twoclass(KFC_classifier, two_class_classifier, Xref, Xtest, blocks)
        
        # make full conformal e-values

        e_values_KFC = e_function_static(adjustment*alpha_fdr, real_score_matrix, cal_weights, test_weights)
        e_values_two = e_function_static(adjustment*alpha_fdr, two_class_score_matrix, cal_weights, test_weights)
        
        e_rej = eBH(e_values_KFC, alpha_fdr)
        e_rej_two_class = eBH(e_values_two, alpha_fdr)
        
        power_KFC.append(compute_power(e_rej, nonzero))
        power_two_class.append(compute_power(e_rej_two_class, nonzero))
        
        fdp_KFC.append(compute_fdp(e_rej, nulls))
        fdp_two_class.append(compute_fdp(e_rej_two_class, nulls))
        print(f'{exp}/{n_exp} Done')
        
    print(f'Finished {amp}, Power KFC: {np.mean(power_KFC)}, Power Two-Class: {np.mean(power_two_class)}')
    print(f'FDR_KFC: {np.mean(fdp_KFC)}, Two_Class FDR: {np.mean(fdp_two_class)}')
    results_power['KFC'].append(power_KFC)
    results_power['Two-class'].append(power_two_class)

#HYPERPARAMETERS 
amp = 3.0
props_outliers = np.arange(0.1,0.6,0.1)

for prop_outliers in props_outliers:
    power_KFC = []
    power_two_class = []
    
    fdp_KFC = []
    fdp_two_class = []
    
    for exp in range(n_exp):
        np.random.seed(exp)

        #We repeat the same structure of the experiement
        Wset = np.random.uniform(size=(50,50)) * 6 - 3
        n_outliers = int(np.ceil(m * prop_outliers))

        # generate null and alternative hypotheses
        perm = np.random.permutation(m) 
        nonzero = np.sort(perm[:n_outliers])
        nulls = np.sort(perm[n_outliers:])

        # ---------------- UNWEIGHTED -----------------------

        Xref = gen_data(Wset, n, 1)

        # don't generate test dataset under covariate shift
        Xtest0 = gen_data(Wset, m-n_outliers, 1)    # inliers 
        Xtest1 = gen_data(Wset, n_outliers, amp)    # outliers

        # weights are just ones vector
        test_weights = np.ones(m)
        cal_weights = np.ones(n)

        Xtest = np.zeros((m, Xtest0.shape[1]))
        Xtest[nonzero,] = Xtest1
        Xtest[nulls,] = Xtest0

        # ---------------------------------------------------

        KFC_classifier = IsolationForest(max_samples=1.0, n_estimators=50)  # initialize base model
        two_class_classifier = LogisticRegression()

        # test blocks used in combination with reference set for training 
        blocks = np.array_split(np.arange(m), K_blocks)    # split the D_test into K_blocks blocks
        
        # score matrix making subroutine
        real_score_matrix = subroutine(KFC_classifier, Xref, Xtest, blocks)    # original
        two_class_score_matrix = subroutine_with_twoclass(KFC_classifier, two_class_classifier, Xref, Xtest, blocks)
        
        # make full conformal e-values

        e_values_KFC = e_function_static(adjustment*alpha_fdr, real_score_matrix, cal_weights, test_weights)
        e_values_two = e_function_static(adjustment*alpha_fdr, two_class_score_matrix, cal_weights, test_weights)
        
        e_rej = eBH(e_values_KFC, alpha_fdr)
        e_rej_two_class = eBH(e_values_two, alpha_fdr)
        
        power_KFC.append(compute_power(e_rej, nonzero))
        power_two_class.append(compute_power(e_rej_two_class, nonzero))
        
        fdp_KFC.append(compute_fdp(e_rej, nulls))
        fdp_two_class.append(compute_fdp(e_rej_two_class, nulls))
        print(f'{exp}/{n_exp} Done')
        
    print(f'Finished {amp}, Power KFC: {np.mean(power_KFC)}, Power Two-Class: {np.mean(power_two_class)}')
    print(f'FDR_KFC: {np.mean(fdp_KFC)}, Two_Class FDR: {np.mean(fdp_two_class)}')
    results_fdr['KFC'].append(fdp_KFC)
    results_fdr['Two-class'].append(fdp_two_class)

FDR_KFC = []
err_KFC = []

FDR_family = []
err_family = []

power_KFC = []
err_power_KFC = []

power_family = []
err_power_family = []

#Saving the information
for idx, pi in enumerate(props_outliers):
    FDR_KFC.append(np.mean(results_fdr['KFC'][idx]))
    FDR_family.append(np.mean(results_fdr['Family'][idx]))
    err_KFC.append(np.std(results_fdr['KFC'][idx]))
    err_family.append(np.std(results_fdr['Family'][idx]))
for idx, amp in enumerate(amps):
    err_power_KFC.append(np.std(results_power['KFC'][idx]))
    err_power_family.append(np.std(results_power['Family'][idx]))
    power_KFC.append(np.mean(results_power['KFC'][idx]))
    power_family.append(np.mean(results_power['Family'][idx]))

#Save the dataframes
df_fdr = pd.DataFrame({
    'props_outliers': props_outliers,
    'FDR_KFC': FDR_KFC,
    'FDR_family': FDR_family,
    'err_KFC': err_KFC,
    'err_family': err_family
})
df_power = pd.DataFrame({
    'amps': amps,
    'power_KFC': power_KFC,
    'power_family': power_family,
    'err_power_KFC': err_power_KFC,
    'err_power_family': err_power_family
})

df_combined = pd.concat([df_fdr, df_power], axis=1)
df_combined.to_csv('simulation_results/two_class_simulation_results.csv', index=False)