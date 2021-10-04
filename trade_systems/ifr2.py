from __future__ import absolute_import

import MetaTrader5 as mt5
from sklearn.svm import SVR
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def RSI(data, period, applied_price='<CLOSE>'):
    dataframe = pd.DataFrame(data)

    delta = dataframe[applied_price].diff(1)
    delta = delta.dropna()

    up = delta.copy()
    down = delta.copy()

    up[up < 0] = 0
    down[down > 0] = 0
    
    dataframe['<UP>'] = up
    dataframe['<DOWN>'] = down

    avg_gain = dataframe['<UP>'].rolling(window=period).mean()
    avg_loss = abs(dataframe['<DOWN>'].rolling(window=period).mean())

    RS = avg_gain/avg_loss
    
    RSI = 100.0 - (100.0/(1.0 + RS))

    dataframe['<RSI>'] = RSI

    return dataframe


class IFR2Strategy(object):

    def __init__(self, server):
        self._nivel_aumento = 0
        self._server = server

    def _trade_logic(self):
        pass

    def _buy(self, symbol, volume, tp, sl):
        request = {}

    def _sell(self, symbol, volume, tp, sl):
        request = {}

    def _aumenta_mao(self):
        pass


class MT5DataFeedback(object):
    def __init__(self):
        pass

    def _get_ohlc(self):
        pass


def main():
    data = pd.read_csv('database/NEW_OHLC_B3SA3_BOV_T.csv', sep=',')

    data = RSI(data, 2, '<CLOSE>')

    train_index = int(data.shape[0] * (1 - 0.2))

    X_train, X_test = data[['<UP>', '<DOWN>', '<CLOSE>']][:train_index], data[['<UP>', '<DOWN>', '<CLOSE>']][train_index:] 
    y_train, y_test = data['<RSI>'][:train_index], data['<RSI>'][train_index:]

    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    y_train = y_train.fillna(0)
    y_test = y_test.fillna(0)

    model = SVR(kernel='linear', verbose=True)

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(model.score(y_test, y_pred))


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('CTRL+C Pressed')