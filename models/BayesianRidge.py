import pandas as pd
import numpy as np
import tensorflow as tf

data = pd.read_csv('database/OHLC_BBDC4_BOV_T.csv', sep=',')
data.set_index('<DATE>_<TIME>', drop=True, inplace=True)

def build_model():
    pass

from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
