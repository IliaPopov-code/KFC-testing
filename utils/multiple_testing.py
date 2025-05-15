import numpy as np
import pandas as pd

import scipy.stats as ss

import copy
from collections import *

#####
def base_e_function(alpha, scores, n):
    '''
    Computes e-values, while maintaining .
        
        Parameters:
            scores : non-conformity scores.
            n : number of training points.
            alpha : The level at which to control FDR.
            
        Returns:
            e_values : Return the list of e-values.
        
    '''

    train = scores[:n]
    test = scores[n:]
    m = len(scores) - n 
    
    # Get the sorted unique candidate thresholds from V.
    candidate_t = np.sort(np.unique(scores))
    
    T = None
    # Iterate over candidate thresholds (from smallest to largest) to find the infimum T.
    for t in candidate_t:
        count_train = np.sum(train >= t)
        count_test = np.sum(test >= t)
        # Use max(1, count_alt) to avoid division by zero.
        ratio = (m / (n + 1)) * (1 + count_train) / max(1, count_test)
        if ratio <= alpha:
            T = t
            break

    # If no threshold satisfies the condition, set T to infinity.
    if T is None:
        T = np.inf

    # Compute the denominator for the e-value formula.
    count_at_T = np.sum(train >= T)
    # Compute e-values for each alternative.
    e_values = (n + 1) * (test >= T).astype(float) / (1 + count_at_T)
    
    return e_values

def e_function_static(alpha, score_matrix, cal_weights, test_weights): 

    '''
    Computes a set of weighted e-values.
        
        Parameters:
            score_matrix : Matrix of non-conformity scores.
            cal/test_weights : Weights assigned to calibration/test scores.
            alpha : The level at which to control FDR.
            
        Returns:
            e : Outputs a list of e-values.
        
    '''

    m, total = score_matrix.shape
    n = total - m

    sum_cal_weights = sum(cal_weights)
    # denom = test_weights + sum_cal_weights    # this is a m-dim array 

    T_array = np.ones(m) * np.inf
    e = np.zeros(m)

    range_list = list(range(m)) 

    for j in range(m):
        # unpack the jth set of leave-one-out scores
        cal_scores = score_matrix[j,:n]
        test_scores = score_matrix[j,n:]

        sum_weights_j = (sum_cal_weights+test_weights[j])

        combined_weights = np.concatenate((test_weights, cal_weights))
        combined_scores = np.concatenate((test_scores, cal_scores))
        combined_memberships = np.concatenate((np.ones(m), np.zeros(n)))

        inds = np.argsort(-1 * combined_scores)

        ordered_weights = combined_weights[inds]  
        cal_bools = (combined_memberships[inds] == 0)    # determine membership of the ordered scores
        # cal_weights_ordered = ordered_weights * (combined_memberships[inds] == 0)    # only keep the ordered weights belonging to the reference set

        test_seq = np.cumsum(combined_memberships[inds] == 1)
        cal_seq = np.cumsum( cal_bools * ordered_weights )    # this ends up being a np.cumsum over weighted indicators
        test_seq[test_seq == 0] = 1

        hat_fdrs = (cal_seq + test_weights[j]) / test_seq * m / sum_weights_j

        # find the j-specific threshold
        if np.any(hat_fdrs <= alpha):
            T_j = combined_scores[inds[np.where(hat_fdrs <= alpha)[0].max()]]
            # if T == 0:
            #     T = np.min([W > 0])
        else:
            T_j = max(combined_scores) + 1

        T_array[j] = T_j

        denom_j = test_weights[j] + sum( (cal_scores >= T_j) * cal_weights )
        e_j = sum_weights_j * (test_scores[j] >= T_j) / denom_j
        e[j] = e_j 
    return e

def eBH(e, alpha):
    '''
    Runs the eBH procedure to control FDR at given level alpha.
        
        Parameters:
            e : The e-values on which to run the procedure.
            alpha : The level at which to control FDR.
            
        Returns:
            rej: The rejection set; the selected indices that have rejected nulls.
        
    '''
    m = len(e)
    e_sort = np.sort(e)[::-1] # descending order
    khat = 0
    for k in range(m,0,-1):
        if e_sort[k-1] >= m/(alpha*k):
            khat = k
            break
    if khat == 0:
        return np.array([])
    else:
        return np.nonzero(np.array(e) >= m/(alpha*khat))[0]
    
def eBH_infty(e, alpha, idx=None):
    '''
    Given e-values e and for each index j, runs the eBH procedure on e' at level alpha, where
    e' is e with e_j replaced by infinity. 
        
        Parameters:
            e : The e-values on which to run the procedure.
            alpha : The level at which to run eBH.
            idx : (Optional) The specific index at which to run the eBH_infty procedure.
            
        Returns:
            n_rej: A vector such that the jth component contains |eBH(e', alpha)|, where e' is e with e_j replaced by infinity.
        
    '''  
    m = len(e)
    if (idx != None):
        # eBH_infty for a specific idx
        e_copy = copy.deepcopy(e)
        e_copy[idx] = np.inf
        return len(eBH(e_copy, alpha))
    
    e_sort = np.sort(e)[::-1] # descending order
    ranks = m - np.argsort(e).argsort() # ranks (1, 2, ..., m)
    
    # initial eBH
    khat = 0
    for k in range(m,0,-1):
        if e_sort[k-1] >= m/(alpha*k):
            khat = k
            break
    
    # now, solve for what happens in the case where e_j --> infty
    S = 1+np.nonzero(   e_sort >= m/(alpha * (np.array(range(m))+2)  ))[0]    # {k : e_(k) >= m/(a(k+1))}; add 1 because it's zero indexed
    s_hat_list = 1+np.array([max(S[S<=rank-1], default=0) for rank in ranks])  # list[j] = max in S less than rank(e_j), +1 for e_j=inf 
    
    # if khat >= rank, then regular eBH threshold is preserved
    n_rej = np.where(khat >= ranks, khat*np.ones(m), s_hat_list) # list[j] = R_alpha^{e_j-->infty)    
    return n_rej
    
def pBH(p, alpha):
    '''
    Runs the BH procedure to control FDR at given level alpha.
        
        Parameters:
            p : The p-values on which to run the procedure.
            alpha : The level at which to control FDR.
            
        Returns:
            rej: The rejection set; the selected indices that have rejected nulls.
        
    '''
    
    # scipy_rej = np.nonzero(sp.stats.false_discovery_control(ps=p) <= alpha)[0]
    # return scipy_rej
    
    m = len(p)
    p_sort = np.sort(p) # ascending order
    khat = 0
    for k in range(m, 0, -1):
        if (p_sort[k-1] * m / k) <= alpha:
            khat = k
            break
    if khat == 0:
        return np.array([])
    else:
        return np.nonzero(np.array(p) <= (alpha*khat/m))[0]
    
def pBY(p, alpha):
    '''
    Runs the BY procedure to control FDR at given level alpha.
        
        Parameters:
            p : The p-values on which to run the procedure.
            alpha : The level at which to control FDR.
            
        Returns:
            rej: The rejection set; the selected indices that have rejected nulls.
        
    '''
    
    rej = np.nonzero(ss.false_discovery_control(ps=p, method='by') <= alpha)[0]
    return rej
    
#####

# "inverse" operations
def eBH_min_alpha(e):
    m = len(e)
    e_sort = np.sort(e)[::-1]
    # return the minimum value of m/(ke_k), which is the min alpha needed for >0 discoveries
    return min(m / (e_sort * np.array(range(1,1+m))))  

#####

# evaluation operations
def evaluate(rejections, H_1):
    """
    Given the rejections and true nonnulls (H_1), returns the FDP and power of the procedure.
    """
    m_1 = len(H_1)
    R = len(rejections)
    
    if R == 0:
        return {'fdp': 0, 'power': 0}
    
    td = len(set(rejections) & set(H_1))
    return {'fdp': 1 - td/max(R,1), 'power': td/max(m_1,1)}