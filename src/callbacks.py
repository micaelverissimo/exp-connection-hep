__all__ = ['sp_index']

import numpy as np

from loguru import logger
from sklearn.metrics import roc_curve
from tensorflow.keras.callbacks import Callback



class sp_index(Callback):
    """
    A custom Keras callback to monitor and optimize the SP metric during training.
    The SP metric is computed based on the ROC curve, and this callback allows for
    early stopping based on SP, saving the best model weights, and logging the progress.
    Attributes:
        validation_data (tuple): A tuple containing validation features and labels.
        verbose (bool): If True, logs detailed information about SP at the end of each epoch.
        save_the_best (bool): If True, saves the model weights corresponding to the best SP value.
        patience (int or bool): Number of epochs to wait for improvement in SP before stopping training.
        count (int): Counter for epochs without improvement in SP.
        __best_sp (float): The best SP value observed during training.
        __best_weights (list): The model weights corresponding to the best SP value.
    Methods:
        on_epoch_end(epoch, logs):
            Computes the SP metric at the end of each epoch, logs the results, and
            optionally stops training if SP does not improve for a specified number of epochs.
        on_train_end(logs):
            Reloads the best model weights (if `save_the_best` is True) at the end of training.
    """

    def __init__(self,  validation_data, verbose=False,
                        save_the_best=False, 
                        patience=False):

        super(Callback, self).__init__()
        self.verbose = verbose
        self.patience = patience
        self.save_the_best = save_the_best

        self.count = 0
        self.__best_sp = 0.0
        self.__best_weights = None
        self.validation_data = validation_data



    def on_epoch_end(self, epoch, logs={}):

        y_true = self.validation_data[1]
        y_hat = self.model.predict(self.validation_data[0],batch_size=1024, verbose=0).ravel()

        # Computes SP
        fa, pd, thresholds = roc_curve(y_true, y_hat)
        sp = np.sqrt(  np.sqrt(pd*(1-fa)) * (0.5*(pd+(1-fa)))  )

        knee = np.argmax(sp)
        logs['max_sp_val'] = sp[knee]
        logs['max_sp_fa_val'] = fa[knee]
        logs['max_sp_pd_val'] = pd[knee]
    
        if self.verbose:
            logger.info(f"val_sp: {sp[knee]:.4f} (fa:{fa[knee]:.4f}, pd:{pd[knee]:.4f}), patience: {self.count}")

        if round(sp[knee],4) > round(self.__best_sp,4):
            self.__best_sp = sp[knee]
            if self.save_the_best:
                if self.verbose:
                    logger.info('save the best configuration here...' )
                self.__best_weights =  self.model.get_weights()
                logs['max_sp_best_epoch_val'] = epoch
            self.count = 0
        else:
            self.count += 1

        if self.count > self.patience:
            if self.verbose:
                logger.info('Stopping the Training by SP...')
            self.model.stop_training = True



    def on_train_end(self, logs={}):

        if self.save_the_best:
            if self.verbose:
                logger.info('Reload the best configuration into the current model...')
            try:
                self.model.set_weights( self.__best_weights )
            except:
                logger.fatal("Its not possible to set the weights. abort")

