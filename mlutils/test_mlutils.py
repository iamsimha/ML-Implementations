from __future__ import division
from sklearn.metrics import roc_auc_score
import numpy as np
import mlutils

np.random.seed(100)

def test_roc_score():
	"""Verify that my implementation of auc_score agrees with sklearns implementation upto some order"""
	true_values = np.array(np.random.choice([0, 1], size=(1000,), p=[1./3, 2./3]))
	predicted_values = np.array(np.random.uniform(0, 1, 1000))
	assert (roc_auc_score(true_values, predicted_values) - mlutils.auc_score(true_values, predicted_values))/roc_auc_score(true_values, predicted_values) < 0.001