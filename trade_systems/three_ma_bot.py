import MetaTrader5 as mt5
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time
import pytz
from datetime import datetime

plt.style.use('fivethirtyeight')

dataframe = pd.read_csv('database\IBOV_D1.csv', sep='\t')

# dataframe['Datetime'] = dataframe['Date'] + ' ' + dataframe['Time']

dataframe = dataframe.set_index(pd.DatetimeIndex(dataframe['<DATE>'].values))

# plt.figure(figsize=(12.2, 4.5))
# plt.title('Close price WIN$N', fontsize=18)
# plt.plot(dataframe['Close'])
# plt.xlabel('Date', fontsize=16)
# plt.ylabel('Close price', fontsize=16)

dataframe_2020 = dataframe.loc['2020.01.01':]

short_ema = dataframe_2020.Close.ewm(span=5, adjust=False).mean()
middle_ema = dataframe_2020.Close.ewm(span=21, adjust=False).mean()
long_ema = dataframe_2020.Close.ewm(span=63, adjust=False).mean()


# plt.figure(figsize=(20.2, 6.5))
# plt.title('Close price WIN$N', fontsize=18)
# plt.plot(dataframe_2020['Close'], label='close price', color='blue')
# plt.plot(short_ema, label='shot ema', color='red')
# plt.plot(middle_ema, label='middle ema', color='orange')
# plt.plot(long_ema, label='long ema', color='green')
# plt.xlabel('Date', fontsize=16)
# plt.ylabel('Close price', fontsize=16)

df['short'] = short_ema
df['middle'] = middle_ema
df['long'] = long_ema

def buy_sell_function(data):
    buy_list = []
    sell_list = []

'''
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1500)
pd.set_option('mode.chained_assignment', None)

timezone = pytz.timezone('ETC/UTC')

data_today = datetime.today().strftime('%Y-%m-%d')
horario_inicio_mercado = pd.Timestamp(data_today + '-10:00:00')
horario_fechamento_mercado = pd.Timestamp(data_today + '-17:00:00')

mt5.initialize()

ativo = 'WINM21'
mt5.symbol_select(ativo, True)

quantidade = 10

def ordem_compra(ativo, quantidade):
    volume = float(quantidade)
    symbol = ativo
    price = mt5.symbol_info_tick(ativo).ask
    deviation = 0

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": price - 100,
        "tk": price + 500,
        "deviation": deviation,
        "magic": 6185154,
        "comment": "Ordem de compra enviada",
        "time_type": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    result = mt5.order_send(request)
    return result

def ordem_venda(ativo, quantidade):
    volume = float(quantidade)
    symbol = ativo
    price = mt5.symbol_info_tick(ativo).bid
    deviation = 0

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": mt5.ORDER_TYPE_SELL,
        "price": price,
        "sl": price + 100,
        "tk": price - 500,
        "deviation": deviation,
        "magic": 6185154,
        "comment": "Ordem de venda enviada",
        "time_type": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    result = mt5.order_send(request)
    return result

def fechar_posicao(ativo, quantidade, ticket, type_order, magic, deviation):
    if(type_order == 0):
        print("Fechamendo da compra")
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": ativo,
            "volume": quantidade,
            "deviation": deviation,
            "magic": magic,
            "price": mt5.symbol_info_tick(ativo).bid,
            "type": mt5.ORDER_TYPE_SELL,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        result = mt5.order_send(request)
        print(result)

    else:
        print("Fechamendo da venda")
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "position": ticket,
            "symbol": ativo,
            "volume": quantidade,
            "deviation": deviation,
            "magic": magic,
            "price": mt5.symbol_info_tick(ativo).ask,
            "type": mt5.ORDER_TYPE_BUY,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_RETURN,
        }
        result = mt5.order_send(request)
        print(result)

    return result


while True:
    hora_agora = datetime.now()
    to_dentro_horario = hora_agora >= horario_inicio_mercado and hora_agora <= horario_fechamento_mercado
    ordens_abertas = mt5.orders_total()
    posicoes_abertas = mt5.positions_total()

    time.sleep(5)
    if (ordens_abertas == 0 and posicoes_abertas == 0):
        print(ordem_compra(ativo, quantidade))
    else:
        time.sleep(10)
        info_posicoes = mt5.positions_get(symbol=ativo)
        if (len(info_posicoes) > 0):
            df = pd.DataFrame(list(info_posicoes), columns=info_posicoes[0]._asdict().keys())
            df['time'] = pd.to_datetime(df['time'], unit='s')

            fechar_posicao(str(df['symbol'][0]), float(df['volume'][0]), int(df['ticket'][0]), df['type'][0], int(df['magic'][0]), 0)


mt5.shutdown()
'''