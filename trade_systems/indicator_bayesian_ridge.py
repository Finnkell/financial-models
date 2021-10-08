from __future__ import absolute_import

import MetaTrader5 as mt5
from sklearn.linear_model import BayesianRidge
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from time import sleep
import os


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


class MT5DataFeed(object):
    def __init__(self):
        mt5.initialize()

    def _get_ohlc(self):
        return mt5.copy_rates_from('WINV21', mt5.TIMEFRAME_M1, datetime.now(), 100)

    def _get_volume(self):
        return mt5.copy_ticks_from('WINV21', datetime.now(), 100, mt5.COPY_TICKS_TRADE)['volume']


def dia_operar(date_now):

    time_now = date_now

    if time_now.hour >= 16 or time_now.hour < 9:
        return False

    return True


def main():
    data = pd.read_csv('database/OHLC_1SWIN$N_BMF_T.csv', sep=',')

    data = RSI(data, 2, '<CLOSE>')

    data = data.set_index('<DATE>_<TIME>')
    
    y = pd.DataFrame()

    y['<RSI>'] = data['<RSI>']
    
    data = data.drop(['<TICK>', '<UP>', '<DOWN>', '<RSI>'], axis=1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data)
    data_scaled = pd.DataFrame(scaler.transform(data)).rename(columns={0: '<OPEN>', 1: '<HIGH>', 2: '<LOW>', 3: '<CLOSE>', 4: '<VOL>'}).fillna(0)

    scaler_y = MinMaxScaler(feature_range=(0, 1))
    scaler_y.fit(y)
    y_scaled = pd.DataFrame(scaler_y.transform(y)).rename(columns={0: '<RSI>'}).fillna(0)

    train_index = int(data_scaled.shape[0] * (1 - 0.2))

    X_train, X_test = data_scaled[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']][:train_index].fillna(0), data_scaled[['<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>']][train_index:len(data_scaled)].fillna(0)
    y_train, y_test = y_scaled[['<RSI>']][:train_index].fillna(0), y_scaled[['<RSI>']][train_index:len(y_scaled)].fillna(0)

    model = BayesianRidge()

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    from sklearn.metrics import mean_squared_error

    print(f'Model score: {mean_squared_error(y_test, y_pred)} RMSE')

    server = MT5DataFeed()

    have_position = False
    indicator = []
    saved = False
    last_ohlc = [(1,)]

    while True:

        while dia_operar(datetime.now()):

            saved = False

            ohlc = server._get_ohlc()
            volume = server._get_volume()

            if len(volume) == 0:
                pass
            else:
                if last_ohlc[0][0] != ohlc[0][0]:
                    scaled = scaler.transform([(ohlc[0][1], ohlc[0][2], ohlc[0][3], ohlc[0][4], volume[-1])])
                    pred = model.predict(scaled)
                    indicator.append(pred)
                    last_ohlc = ohlc
                    pred = scaler_y.inverse_transform(pred.reshape(-1, 1))

                    print(f"{datetime.now().hour}:{datetime.now().minute}:{datetime.now().second} | {pred}")

                    positions = mt5.positions_get(symbol='WINV21')

                    for position in positions:
                        if position.magic == 3333333:
                            have_position = True
                        else:
                            have_position = False

                    if not have_position:

                        if pred[0] >= 70 and pred[0] <= 100:
                            price = mt5.symbol_info('WINV21').bid

                            request = {
                                "action": mt5.TRADE_ACTION_DEAL,
                                "symbol": 'WINV21',
                                "volume": 1.0,
                                "type": mt5.ORDER_TYPE_SELL,
                                "price": price,
                                "sl": price + 100,
                                "tp": price - 200,
                                "deviation": 1,
                                "magic": 3333333,
                                "comment": "python script open",
                                "type_time": mt5.ORDER_TIME_GTC,
                                "type_filling": mt5.ORDER_FILLING_RETURN,
                            }

                            mt5.order_send(request)

                        elif pred[0] <= 30 and pred[0] >= 0:
                            price = mt5.symbol_info('WINV21').ask

                            request = {
                                "action": mt5.TRADE_ACTION_DEAL,
                                "symbol": 'WINV21',
                                "volume": 1.0,
                                "type": mt5.ORDER_TYPE_BUY,
                                "price": price,
                                "sl": price - 100,
                                "tp": price + 200,
                                "deviation": 1,
                                "magic": 3333333,
                                "comment": "python script open",
                                "type_time": mt5.ORDER_TIME_GTC,
                                "type_filling": mt5.ORDER_FILLING_RETURN,
                            }

                            mt5.order_send(request)
                else:
                    pass

        if (os.path.exists(f'plots/bayesian/indicator_{datetime.now().day}_{datetime.now().month}_{datetime.now().day}')):
            pass
        elif not saved and len(indicator) > 0:
            indicator = pd.DataFrame(indicator)
            indicator[0].plot()
            plt.savefig(f'plots/bayesian/indicator_{datetime.now().day}_{datetime.now().month}_{datetime.now().day}.png')
            saved = True
        else:
            pass


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('CTRL+C Pressed')