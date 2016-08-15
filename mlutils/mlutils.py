from __future__ import division
import numpy as np
from scipy.integrate import trapz, simps


def auc_score(true_values, predicted_values):
	"""true_values : numpy array, with values in the range 0 to 1
		predicted_values : numpy array, with values in the range 0 to 1
	"""
	thresholds = np.linspace(0, 1, 100)

	fpr_array, tpr_array = [], []
	_true_values = np.copy(true_values)
	_predicted_values = np.copy(predicted_values)
	for threshold in thresholds:
		# Binary Thresholding
		_true_values[_true_values < threshold] = 0
		_true_values[_true_values > threshold] = 1
		_predicted_values[_predicted_values < threshold] = 0
		_predicted_values[_predicted_values > threshold] = 1

		# compute False positive rate
		fp = np.sum((_true_values == 0) & (_predicted_values == 1))
		tn = np.sum((_true_values == 0) & (_predicted_values == 0))
		fpr = fp / (fp + tn)

		# compute True positive rate
		tp = np.sum((_true_values == 1) & (_predicted_values == 1))
		fn = np.sum((_true_values == 1) & (_predicted_values == 0))
		tpr = tp / (tp + fn)

		fpr_array.append(fpr)
		tpr_array.append(tpr)
		_true_values = np.copy(true_values)
		_predicted_values = np.copy(predicted_values)

	# Reorder x and y points, so that trapezoidal can be applied correctly.
	# Reording code inspired by sklearn implementation of roc_auc_score function.
	fpr_array = np.array(fpr_array)
	tpr_array = np.array(tpr_array)
	order = np.lexsort((tpr_array, fpr_array))
	fpr_array, tpr_array = fpr_array[order], tpr_array[order]

	# numerically integrate the tpr-fpr curve using trapezoidal rule
	return trapz(np.array(tpr_array), np.array(fpr_array))