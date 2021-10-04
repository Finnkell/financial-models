from __future__ import absolute_import

import MetaTrader5 as mt5
from sklearn.linear_model import ElasticNet
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

    avg_gain = dataframe['<UP>'].rolling(window=9).mean()
    avg_loss = abs(dataframe['<DOWN>'].rolling(window=9).mean())

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

    data['<RSI>'][:50].plot()
    plt.show()


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('CTRL+C Pressed')