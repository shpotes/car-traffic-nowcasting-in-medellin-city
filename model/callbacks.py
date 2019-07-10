#!/usr/bin/env python
from functools import reduce
from tensorflow.keras.callbacks import ReduceLROnPlateau

def reduceLR(model):
    return ReduceLROnPlateau(patience=5, verbose=1, mode='auto')

def tensorboard(model):
    return 
