
from .multiple_testing import eBH, base_e_function
import numpy as np


def find_extremal_set(rej_sets, maximum=True):
    """
    Find the index of the set of the greatest/smallest size

    Parameters:
        rej_sets : collection of the rejection sets.
        maximum (optional) : a boolean to determine whether we want a maximal or minimal set.
        
    Returns:
        idx: index of the maximal/minimal set.
    """

    if maximum:
        idx = max(range(len(rej_sets)), key=lambda i: len(rej_sets[i]))
    else:
        idx = min(range(len(rej_sets)), key=lambda i: len(rej_sets[i]))
    return idx
    

def rej_classifier(classifier, Xtrain, Xtest, block, alpha_fdr, adj = True):
    """
    Produces the rejection set for a given classifier

    Parameters:
        classifier : one-class classifier object.
        Xtrain : training data.
        Xtest : testing data.
        block : block size
        alpha_fdr : FDR bound
        adj (optional): whether we want to change the alpha of eBH and alpha of KFC
    
    Returns:
        e_rej: outputs the rejection list
    """
    if adj:
        adjustment= 1/(1+alpha_fdr)
    else:
        adjustment = 1
    
    m = Xtest.shape[0]    
    n = Xtrain.shape[0]

    Xtrain = np.vstack((Xtrain, Xtest[block]))
    all_but_block = np.setdiff1d(range(m), block)
    Xtest = Xtest[all_but_block]
    
    #We make an augmented matrix of all values
    Xaugmented = np.vstack((Xtrain, Xtest))
    classifier.fit(Xtrain)

    # score only test samples
    score_samples = -1*classifier.score_samples(Xaugmented).flatten()
    e_values = base_e_function(adjustment*alpha_fdr, score_samples, n)
    e_rej = eBH(e_values, alpha_fdr)
    return e_rej