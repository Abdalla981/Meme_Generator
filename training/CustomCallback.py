import numpy as np
from scipy.stats import gmean
from tensorflow.keras.callbacks import Callback

class CustomCallback(Callback):
    def __init__(self, model_path, patience=0):
        super(CustomCallback, self).__init__()
        self.model_path = model_path
        self.patience = patience
        self.best_weights = None
        
    def on_train_begin(self, logs=None):
        # The number of epoch it has waited when loss is no longer minimum.
        self.wait = 0
        # The epoch the training stops at.
        self.stopped_epoch = 0
        # Initialize the best as infinity.
        self.best_loss = np.Inf
        self.best_val_loss = np.Inf
        self.best_epoch = 0
    
    def on_epoch_end(self, epoch, logs=None):
        if logs['val_loss'] < self.best_val_loss:
            self.best_loss = logs['loss']
            self.best_val_loss = logs['val_loss']
            self.best_epoch = epoch + 1
            self.wait = 0
            self.best_weights = self.model.get_weights()
        else:
            self.wait += 1
            if self.wait > self.patience:
                self.stopped_epoch = epoch
                self.model.stop_training = True
                print("Restoring model weights from the end of the best epoch.")
                self.model.set_weights(self.best_weights)
                
    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))
        self.model.save(f'{self.model_path}-ep{self.best_epoch:03d}-loss{self.best_loss:.3f}-val_loss{self.best_val_loss:.3f}.h5')
        