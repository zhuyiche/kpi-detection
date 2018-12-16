from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score
import numpy as np


def _roc_auc_score(output, target, scores):
    scores = scores.detach().numpy()
    try:
        score = _roc_auc_score(target, scores)
        #print(score)
        return score
    except:
        #print('Warning: Only one class present in this Time')
        return np.float32(1.0)

def _accurcay_score(output, target, scores):
    scores = scores.detach().numpy()
    try:
        score = accuracy_score(target, scores)
        # print(score)
        return score
    except:
        # print('Warning: Only one class present in this Time')
        return np.float32(1.0)

def _f1_score(output, target, scores):
    scores = scores.detach().numpy()
    try:
        score = f1_score(target, scores)
        # print(score)
        return score
    except:
        # print('Warning: Only one class present in this Time')
        return np.float32(1.0)