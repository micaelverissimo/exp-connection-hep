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


#
# Use this class to decorate the history with the reference values configured by the user 
#
class Reference:

  #
  # Constructor
  #
  def __init__( self , refs):
    
    self.__references = collections.OrderedDict()

    for key, ref in refs.items():

      pd = [ref['det']['passed'] , ref['det']['total']]
      pd = [pd[0]/pd[1], pd[0], pd[1]]
      fa = [ref['fake']['passed'] , ref['fake']['total']]
      fa = [fa[0]/fa[1], fa[0], fa[1]]
      logger.info('%s (pd=%1.2f, fa=%1.2f, sp=%1.2f)', key, pd[0]*100, fa[0]*100, sp_func(pd[0],fa[0])*100 )
      self.__references[key] = {'pd':pd, 'fa':fa, 'sp':sp_func(pd[0],fa[0])}

 

  #
  # decorate the history after the training phase
  #
  def __call__( self, history, kw ):
    

    model            = kw["model"]
    x_train, y_train = kw["data"]
    x_val , y_val    = kw["data_val"]

    y_pred     = model.predict( x_train, batch_size = 1024, verbose=0 )
    y_pred_val = model.predict( x_val  , batch_size = 1024, verbose=0 )

    # get vectors for operation mode (train+val)
    y_pred_operation = np.concatenate( (y_pred, y_pred_val), axis=0)
    y_operation = np.concatenate((y_train,y_val), axis=0)


    train_total = len(y_train)
    val_total = len(y_val)

    # Here, the threshold is variable and the best values will
    # be setted by the max sp value found in hte roc curve
    # Training
    fa, pd, thresholds = roc_curve(y_train, y_pred)
    sp = np.sqrt(  np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa)))  )

    # Validation
    fa_val, pd_val, thresholds_val = roc_curve(y_val, y_pred_val)
    sp_val = np.sqrt(  np.sqrt(pd_val*(1-fa_val)) * (0.5*(pd_val+(1-fa_val)))  )

    # Operation
    fa_op, pd_op, thresholds_op = roc_curve(y_operation, y_pred_operation)
    sp_op = np.sqrt(  np.sqrt(pd_op*(1-fa_op)) * (0.5*(pd_op+(1-fa_op)))  )


    history['reference'] = {}

    for key, ref in self.__references.items():
      d = self.calculate( y_train, y_val , y_operation, ref, pd, fa, sp, thresholds, pd_val, fa_val, sp_val, thresholds_val, pd_op,fa_op,sp_op,thresholds_op )
      logger.info( "          : %s", key )
      logger.info( "Reference : [Pd: %1.4f] , Fa: %1.4f and SP: %1.4f ", ref['pd'][0]*100, ref['fa'][0]*100, ref['sp']*100 )
      logger.info( "Train     : [Pd: %1.4f] , Fa: %1.4f and SP: %1.4f ", d['pd'][0]*100, d['fa'][0]*100, d['sp']*100 )
      logger.info( "Validation: [Pd: %1.4f] , Fa: %1.4f and SP: %1.4f ", d['pd_val'][0]*100, d['fa_val'][0]*100, d['sp_val']*100 )
      logger.info( "Operation : [Pd: %1.4f] , Fa: %1.4f and SP: %1.4f ", d['pd_op'][0]*100, d['fa_op'][0]*100, d['sp_op']*100 )
      history['reference'][key] = d




  #
  # Calculate sp, pd and fake given a reference
  # 
  def calculate( self, y_train, y_val , y_op, 
                 ref, pd, fa, sp, thresholds, 
                 pd_val,fa_val, sp_val, thresholds_val, 
                 pd_op,fa_op, sp_op, thresholds_op ):

    d = {}
    def closest( values , ref ):
      index = np.abs(values-ref)
      index = index.argmin()
      return values[index], index


    # Check the reference counts
    op_total = len(y_op[y_op==1])
    if ref['pd'][2] !=  op_total:
      ref['pd'][2] = op_total
      ref['pd'][1] = int(ref['pd'][0]*op_total)

    # Check the reference counts
    op_total = len(y_op[y_op!=1])
    if ref['fa'][2] !=  op_total:
      ref['fa'][2] = op_total
      ref['fa'][1] = int(ref['fa'][0]*op_total)


    d['pd_ref'] = ref['pd']
    d['fa_ref'] = ref['fa']
    d['sp_ref'] = ref['sp']
    #d['reference'] = ref['reference']


    # Train
    _, index = closest( pd, ref['pd'][0] )
    train_total = len(y_train[y_train==1])
    d['pd'] = ( pd[index],  int(train_total*float(pd[index])),train_total)
    train_total = len(y_train[y_train!=1])
    d['fa'] = ( fa[index],  int(train_total*float(fa[index])),train_total)
    d['sp'] = sp_func(d['pd'][0], d['fa'][0])
    d['threshold'] = thresholds[index]


    # Validation
    _, index = closest( pd_val, ref['pd'][0] )
    val_total = len(y_val[y_val==1])
    d['pd_val'] = ( pd_val[index],  int(val_total*float(pd_val[index])),val_total)
    val_total = len(y_val[y_val!=1])
    d['fa_val'] = ( fa_val[index],  int(val_total*float(fa_val[index])),val_total)
    d['sp_val'] = sp_func(d['pd_val'][0], d['fa_val'][0])
    d['threshold_val'] = thresholds_val[index]


    # Train + Validation
    _, index = closest( pd_op, ref['pd'][0] )
    op_total = len(y_op[y_op==1])
    d['pd_op'] = ( pd_op[index],  int(op_total*float(pd_op[index])),op_total)
    op_total = len(y_op[y_op!=1])
    d['fa_op'] = ( fa_op[index],  int(op_total*float(fa_op[index])),op_total)
    d['sp_op'] = sp_func(d['pd_op'][0], d['fa_op'][0])
    d['threshold_op'] = thresholds_op[index]

    return d