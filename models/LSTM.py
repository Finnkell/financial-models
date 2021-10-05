import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import time
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from silence_tensorflow import silence_tensorflow

silence_tensorflow()

plt.style.use(style='seaborn')

dataframe = pd.read_csv('database/NEW_OHLC_B3SA3_BOV_T.csv', sep=',')

dataframe.set_index('<DATE>_<TIME>', drop=True, inplace=True)

dataframe['<RETURNS>'] = dataframe['<CLOSE>'].pct_change().fillna(0)
dataframe['<LOG_RETURNS>'] = np.log(1 + dataframe['<RETURNS>'])

X = dataframe[['<CLOSE>', '<LOG_RETURNS>']].values

scaler = MinMaxScaler(feature_range=(0, 1)).fit(X)
X_scaled = scaler.transform(X)

y = [x[0] for x in X_scaled]

split = int(len(X) * (1 - 0.2))

X_train = X_scaled[:split]
X_test = X_scaled[split:len(X_scaled)]
y_train = y[:split]
y_test = y[split:len(y)]

assert len(X_train) == len(y_train)
assert len(X_test) == len(y_test)

n = 5
x_train_list = []
x_test_list = []
y_train_list = []
y_test_list = []


for i in range(n, len(X_train)):
    x_train_list.append(X_train[i - n : i, : X_train.shape[1]])
    y_train_list.append(y_train[i])

for i in range(n, len(X_test)):
    x_test_list.append(X_test[i - n : i, : X_test.shape[1]])
    y_test_list.append(y_test[i])


val = np.array(y_train_list[0])
val = np.c_[val, np.zeros(val.shape)]

scaler.inverse_transform(val)

Xtrain, ytrain = (np.array(x_train_list), np.array(y_train_list))
Xtrain = np.reshape(Xtrain, (Xtrain.shape[0], Xtrain.shape[1], Xtrain.shape[2]))

Xtest, ytest = (np.array(x_test_list), np.array(y_test_list))
Xtest = np.reshape(Xtest, (Xtest.shape[0], Xtest.shape[1], Xtest.shape[2]))


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

model = Sequential()
model.add(LSTM(4, input_shape=(Xtrain.shape[1], Xtrain.shape[2])))
model.add(Dense(26))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')

model.fit(Xtrain, ytrain, epochs=1, validation_data=(Xtest, ytest), batch_size=160, verbose=True)

train_pred = model.predict_on_batch(ytrain)
test_pred = model.predict(ytest)

train_pred = np.c_[train_pred, np.zeros(train_pred.shape)] # concat
test_pred = np.c_[test_pred, np.zeros(test_pred.shape)] # concat

train_pred = scaler.inverse_transform(train_pred)
train_pred = [x[0] for x in train_pred]

test_pred = scaler.inverse_transform(test_pred)
test_pred = [x[0] for x in test_pred]

print(train_pred[:5])
print(test_pred[:5])


from sklearn.metrics import mean_squared_error

train_score = mean_squared_error([x[0][0] for x in Xtrain], train_pred, squared=False)
print(f'Train score: {train_score} RMSE')

test_score = mean_squared_error([x[0][0] for x in Xtest], test_pred, squared=False)
print(f'Test score: {test_score} RMSE')

