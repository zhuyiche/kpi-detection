from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import numpy as np


def _roc_auc_score(output, target, scores):
    scores = scores.detach().numpy()
    try:
        score = roc_auc_score(target, scores)
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
        precision = precision_score(target, scores)
        recall = recall_score(target, scores)
        print('precision: {}, recall: {}'.format(precision, recall))
        score = 2 * (precision * recall) / (precision + recall)
        #score = f1_score(target, scores)
        # print(score)
        return score
    except:
        # print('Warning: Only one class present in this Time')
        return np.float32(1.0)