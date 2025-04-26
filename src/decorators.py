__all__ = ["Summary"]


from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
from sklearn.metrics import mean_squared_error

from loguru import logger
from tensorflow.keras.models import Model, model_from_json
from copy import copy

import numpy as np
import collections
import time
import pandas

def sp_func(pd, fa):
    """
    Calculate the SP metric based on probability of detection (pd) and false alarm (fa).

    Args:
        pd (float): Probability of detection.
        fa (float): False alarm rate.

    Returns:
        float: The calculated SP metric.
    """
    return np.sqrt(np.sqrt(pd * (1 - fa)) * (0.5 * (pd + (1 - fa))))


class Summary:
    """
    A decorator class to enhance the Keras training history with additional metrics and summaries.

    Attributes:
        detailed (bool): If True, includes detailed ROC curves and histograms in the summary.
    """

    def __init__(self, detailed=False):
        """
        Initialize the Summary class.

        Args:
            detailed (bool): Whether to include detailed metrics in the summary.
        """
        self.detailed = detailed

    def __call__(self, history, kw):
        """
        Decorate the Keras training history with additional metrics.

        Args:
            history (dict): The Keras training history dictionary.
            kw (dict): A dictionary containing training data, validation data, and the model.

        Returns:
            None: Updates the history dictionary in place.
        """
        d = {}
        x_train, y_train = kw["data"]
        x_val, y_val = kw["data_val"]
        model = kw["model"]

        # Get the number of events for each set (train/val)
        sgn_total = len(y_train[y_train == 1])
        bkg_total = len(y_train[y_train != 1])
        sgn_total_val = len(y_val[y_val == 1])
        bkg_total_val = len(y_val[y_val != 1])

        logger.info("Starting the train summary...")

        y_pred = model.predict(x_train, batch_size=1024, verbose=0)
        y_pred_val = model.predict(x_val, batch_size=1024, verbose=0)

        # Combine predictions for operational mode
        y_pred_op = np.concatenate((y_pred, y_pred_val), axis=0)
        y_op = np.concatenate((y_train, y_val), axis=0)

        if self.detailed:
            d['rocs'] = {}
            d['hists'] = {}

        # Training metrics
        fa, pd, thresholds = roc_curve(y_train, y_pred)
        sp = sp_func(pd, fa)
        knee = np.argmax(sp)
        threshold = thresholds[knee]

        step = 1e-2
        bins = np.arange(min(y_train), max(y_train) + step, step=step)

        if self.detailed:
            d['rocs']['roc'] = (pd, fa)
            d['hists']['trn_sgn'] = np.histogram(y_pred[y_train == 1], bins=bins)
            d['hists']['trn_bkg'] = np.histogram(y_pred[y_train != 1], bins=bins)

        logger.info(f"Train samples : Prob. det ({pd[knee]:1.4f}), False Alarm ({fa[knee]:1.4f}), SP ({sp[knee]:1.4f})")

        d['max_sp_pd'] = (pd[knee], int(pd[knee] * sgn_total), sgn_total)
        d['max_sp_fa'] = (fa[knee], int(fa[knee] * bkg_total), bkg_total)
        d['max_sp'] = sp[knee]

        # Validation metrics
        fa, pd, thresholds = roc_curve(y_val, y_pred_val)
        sp = sp_func(pd, fa)
        knee = np.argmax(sp)
        threshold = thresholds[knee]

        if self.detailed:
            d['rocs']['roc_val'] = (pd, fa)
            d['hists']['val_sgn'] = np.histogram(y_pred_val[y_val == 1], bins=bins)
            d['hists']['val_bkg'] = np.histogram(y_pred_val[y_val != 1], bins=bins)

        logger.info(f"Validation Samples: Prob. det ({pd[knee]:1.4f}), False Alarm ({fa[knee]:1.4f}), SP ({sp[knee]:1.4f})")

        d['max_sp_pd_val'] = (pd[knee], int(pd[knee] * sgn_total_val), sgn_total_val)
        d['max_sp_fa_val'] = (fa[knee], int(fa[knee] * bkg_total_val), bkg_total_val)
        d['max_sp_val'] = sp[knee]
        d['threshold_val'] = threshold
    
        # Operational metrics
        fa, pd, thresholds = roc_curve(y_op, y_pred_op)
        sp = sp_func(pd, fa)
        knee = np.argmax(sp)
        threshold = thresholds[knee]

        if self.detailed:
            d['rocs']['roc_op'] = (pd, fa)
            d['hists']['op_sgn'] = np.histogram(y_pred_op[y_op == 1], bins=bins)
            d['hists']['op_bkg'] = np.histogram(y_pred_op[y_op != 1], bins=bins)

        logger.info(f"Operation Samples : Prob. det ({pd[knee]:1.4f}), False Alarm ({fa[knee]:1.4f}), SP ({sp[knee]:1.4f})")

        d['threshold_op'] = threshold
        d['max_sp_pd_op'] = (pd[knee], int(pd[knee] * (sgn_total + sgn_total_val)), (sgn_total + sgn_total_val))
        d['max_sp_fa_op'] = (fa[knee], int(fa[knee] * (bkg_total + bkg_total_val)), (bkg_total + bkg_total_val))
        d['max_sp_op'] = sp[knee]

        history['summary'] = d
