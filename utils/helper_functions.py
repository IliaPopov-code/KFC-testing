import numpy as np

def compute_power(rejections, H_1):
    true_d = len(set(rejections) & set(H_1))
    return true_d/max(1, len(H_1))

def compute_fdp(rejections, H_1):
    R = len(rejections)
    true_d = len(set(rejections) & set(H_1))
    return true_d/max(R,1)

def quantile_func(score, scores):
    scores = np.array(scores)
    return np.mean(scores <= score)

def quantile_inverse(q, scores):
    scores = np.array(scores)
    return np.percentile(scores, q * 100)

def logit(x):
    if x == 1:
        x -= 0.001
    return np.log(x / (1 - x))

def logistic(z):
    return 1 / (1 + np.exp(-z))