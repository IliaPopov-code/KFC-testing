import numpy as np

from .helper_functions import quantile_func, logistic, logit, quantile_inverse
from .family_wise import rej_classifier, find_extremal_set
from .multiple_testing import e_function_static, eBH

def compute_beta(n, m, K_blocks):
    """
    Computes the parameter with which we increase quintile

    Parameters:
        n : number of points in training sample.
        m : number of points in the testing sample.
        K_blocks : number of block.
        
    Returns:
        beta: Parameter of increase.
    """
    B_k_size = m//K_blocks
    # Check to avoid division by zero
    if B_k_size == 0:
        raise ValueError("The size of B_k and epsilon must be greater than 0.")
        
    epsilon = 1/( n+ B_k_size)
    result = (n + m) / (n + B_k_size) * np.log((1 - epsilon) / (B_k_size * epsilon))
    return result

def subroutine(classifier, Xref, Xtest, blocks):

    """
    Given a classifier and both the reference and the test dataset, as well as number of blocks outputs the non-conformity scores. 
        
        Parameters:
            classifier : Classification object.
            Xref : Reference dataset.
            Xtest : Testing dataset.
            
        Returns:
            scores: A vector of non-conformity scores
    """

    m = Xtest.shape[0]
    n = Xref.shape[0]
    
    scores = np.zeros(shape=(m,n+m))
    if len(blocks)==1:
        # 1-FC: this is just the default full conformal, without leave-one-out training
        Xaugmented = np.vstack((Xref, Xtest))
        
        # train on the ref+test dataset
        classifier.fit(Xaugmented)
        
        # score samples
        score_samples = classifier.score_samples(Xaugmented).flatten()
    
        # since we only have 1 block, only one distinct row in the conformity score matrix
        scores = np.tile(score_samples, (m, 1))
        
        return -1 * scores
    
    for block_idx, block in enumerate(blocks):
        Xaugmented = np.vstack((Xref, Xtest[block]))

        # train on the augmented dataset
        classifier.fit(Xaugmented)

        # score the samples
        # D_ref first
        score_samples = classifier.score_samples(Xaugmented).flatten()
        scores[block,:n] = score_samples[:n]
        
        # D_test[block] next
        test_scores = np.zeros(m)
        test_scores[block] = score_samples[n:]
        
        # D_test[-block] last
        all_but_block = np.setdiff1d(range(m), block)
        test_scores[all_but_block] = classifier.score_samples(Xtest[all_but_block]).flatten()
        scores[block,n:] = test_scores
        
    return -1 * scores

def subroutine_with_twoclass(base_classifier, twoclass_classifier, Xref, Xtest, blocks, negate_scores=True):

    """
    Given a classifier, two-class calssifier, and both the reference and the test dataset, as well as number of blocks outputs the non-conformity scores. 
        
        Parameters:
            classifier : Classification object.
            two-class classifier : Two-class classification object.
            Xref : Reference dataset.
            Xtest : Testing dataset.
            
        Returns:
            scores: A vector of non-conformity scores
    """

    m = Xtest.shape[0]
    n = Xref.shape[0]
    scores = np.zeros((m, n+m))
    beta = compute_beta(n, m, len(blocks))
    
    if len(blocks) == 1:
        # this is just the default full conformal, without leave-one-out training
        X_train = np.vstack((Xref, Xtest))
        y_train = np.hstack((np.zeros(n), np.ones(m)))
        
        # Train the base classifier on the same combined dataset.
        Xaugmented = X_train  # same as stacking Xref and Xtest
        base_classifier.fit(Xaugmented)
        base_scores = base_classifier.score_samples(Xaugmented).flatten()
        
        # In the K=1 case, the conformity matrix has identical rows.
        scores = np.tile(base_scores, (m, 1))
        return -scores if negate_scores else scores

    for block in blocks:
        # Here we train a two-class classifier
        X_pos = np.vstack((Xref, Xtest[block]))
        y_pos = np.zeros(X_pos.shape[0]) #make a row of 0's
        
        nonblock = np.setdiff1d(np.arange(m), block)
        X_neg = Xtest[nonblock]
        y_neg = np.ones(X_neg.shape[0]) #make a row of 1's
        
        X_train_twoclass = np.vstack((X_pos, X_neg))
        y_train_twoclass = np.hstack((y_pos, y_neg))
    
        twoclass_classifier.fit(X_train_twoclass, y_train_twoclass)
        q_block = twoclass_classifier.predict_proba(Xtest[block])[:, 1] # Get the q-scores
        
        
        # Augment the data and fit the classifier
        Xaug = np.vstack((Xref, Xtest[block]))
        base_classifier.fit(Xaug)
        
        # score the samples
        # D_ref first
        if negate_scores:
            score_samples = -1*base_classifier.score_samples(Xaug).flatten()
        else: 
            score_samples = base_classifier.score_samples(Xaug).flatten()
            
        scores[block,:n] = score_samples[:n]
        
        # D_test[block] next
        updated_scores = []
        block_size = len(score_samples[n:])
        #print(f'Shift of a block {shift_block}')
        
        for idx in range(block_size):
            Q_i = quantile_func(score_samples[n+idx], score_samples)
            shift = beta * max(q_block[idx] - (m - block_size) / (m + n), 0)
            Q_i_adjusted = logistic(logit(Q_i) + shift)
            adjusted_score = quantile_inverse(Q_i_adjusted, score_samples) #compute the adjusted score
            updated_scores.append(adjusted_score)
            
        test_scores = np.zeros(m)
        test_scores[block] = updated_scores
        
        # D_test[-block] last
        all_but_block = np.setdiff1d(range(m), block)
        if negate_scores:
            test_scores[all_but_block] = -1*base_classifier.score_samples(Xtest[all_but_block]).flatten()
        else: 
            test_scores[all_but_block] = base_classifier.score_samples(Xtest[all_but_block]).flatten()
        scores[block,n:] = test_scores

    return scores

def subroutine_familywise(classifier_family, Xref, Xtest, blocks, alpha_fdr):
    m = Xtest.shape[0]
    n = Xref.shape[0]
    adjustment= 0.5
    scores = np.zeros(shape=(m,n+m))

    for block_idx, block in enumerate(blocks):
        rej_sets = []
        for classifier in classifier_family:
            S_j = rej_classifier(classifier, Xref, Xtest, block, alpha_fdr)
            rej_sets.append(S_j)

        #Here we find the most rejections
        idx = find_extremal_set(rej_sets, maximum = True)
        
        #Here we find the least deviations:
        best_classifier = classifier_family[idx]

        Xaugmented = np.vstack((Xref, Xtest[block]))
        best_classifier.fit(Xaugmented)

        # score the samples
        score_samples = best_classifier.score_samples(Xaugmented).flatten()
        scores[block,:n] = score_samples[:n]
        
        # D_test[block] next
        test_scores = np.zeros(m)
        test_scores[block] = score_samples[n:]
        
        # D_test[-block] last
        all_but_block = np.setdiff1d(range(m), block)
        test_scores[all_but_block] = best_classifier.score_samples(Xtest[all_but_block]).flatten()
        scores[block,n:] = test_scores

    e_values_family = e_function_static(adjustment*alpha_fdr, (-1)*scores, np.ones(n), np.ones(m)) #This constant is not optimal
    e_rej = eBH(e_values_family, alpha_fdr)
    return e_rej