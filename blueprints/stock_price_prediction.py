# supervised regresison models
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.neural_network import MLPRegressor

# data analysis and model evaluation
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import f_regression

# deep learning models
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.layers import LSTM
from keras.wrappers.scikit_learn import KerasRegressor

# time series models
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

# data preparation and visualization
import numpy as np
import pandas as pd
import pandas_datareader.data as web
from matplotlib import pyplot
from pandas.plotting import scatter_matrix
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings(action='ignore')

stk_tickers = ['MSFT', 'IBM', 'GOOGL']
ccy_tickers = ['DEXJPUS', 'DEXUSUK']
idx_tickers = ['SP500', 'DJIA', 'VIXCLS']

#  || TICKERS BRASILEIROS ||

# ['PBR', 'CBD', 'VIV', 'EBR', 'BSBR']
# 'PBR'  - Petróleo
# 'CBD'  - Companhia Brasileira de Distribuição
# 'VIV'  - Telefônia Brasil
# 'EBR'  - Centrais Elétricas
# 'BSBR' - Banco Santander

stk_data = web.DataReader(stk_tickers, 'yahoo')
ccy_data = web.DataReader(ccy_tickers, 'fred')
idx_data = web.DataReader(idx_tickers, 'fred')

return_period = 5
Y = np.log(stk_data.loc[:, ('Adj Close', 'MSFT')]).diff(return_period).shift(-return_period)
Y.name = Y.name[-1] + '_pred'

X1 = np.log(stk_data.loc[:, ('Adj Close', ('GOOGL', 'IBM'))]).diff(return_period)
X1.columns = X1.columns.droplevel()
X2 = np.log(ccy_data).diff(return_period)
X3 = np.log(idx_data).diff(return_period)

X4 = pd.concat([np.log(stk_data.loc[:, ('Adj Close', 'MSFT')]).diff(i) for i in [return_period, return_period*3, return_period*6, return_period*12]], axis=1).dropna()
X4.columns = ['MSFT_DT', 'MSFT_3DT', 'MSFT_6DT', 'MSFT_12DT']

X = pd.concat([X1, X2, X3, X4], axis=1)

dataset = pd.concat([Y, X], axis=1).dropna().iloc[::return_period, :]

Y = dataset.loc[:, Y.name]
X = dataset.loc[:, X.columns]

correlation = dataset.corr()
# pyplot.figure(figsize=(15, 15))
# pyplot.title('Correlation Matrix')
# sns.heatmap(correlation, vmax=1, square=True, annot=True)

# pyplot.figure(figsize=(15, 15))
# scatter_matrix(dataset, figsize=(12, 12))

# res = sm.tsa.seasonal_decompose(Y, period=52)
# fig = res.plot()
# fig.set_figheight(8)
# fig.set_figwidth(15)

validation_size = 0.2
train_size = int(len(X) * (1 - validation_size))
X_train, X_test = X[0:train_size], X[train_size:len(X)]
Y_train, Y_test = Y[0:train_size], Y[train_size:len(X)]

num_folds = 10
scoring = 'neg_mean_squared_error'

models = []
models.append(('LR', LinearRegression()))
models.append(('LASSO', Lasso()))
models.append(('EN', ElasticNet()))
models.append(('KNN', KNeighborsRegressor()))
models.append(('CART', DecisionTreeRegressor()))
models.append(('SVR', SVR()))

models.append(('MLP', MLPRegressor()))

models.append(('ABR', AdaBoostRegressor()))
models.append(('GBR', GradientBoostingRegressor()))
models.append(('RFR', RandomForestRegressor()))
models.append(('ETR', ExtraTreesRegressor()))

names = []
kfold_results = []
test_results = []
train_results = []

for name, model in models:
    names.append(name)

    kfold = KFold(n_splits=num_folds, random_state=None)

    cv_results = -1*cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)

    kfold_results.append(cv_results)

    res = model.fit(X_train, Y_train)
    train_result = mean_squared_error(res.predict(X_train), Y_train)
    train_results.append(train_result)

    test_result = mean_squared_error(res.predict(X_test), Y_test)
    test_results.append(test_result)

# fig = pyplot.figure()
# fig.suptitle('KFold Results')
# ax = fig.add_subplot(111)
# pyplot.boxplot(kfold_results)
# ax.set_xticklabels(names)
# fig.set_size_inches(15, 8)

# ARIMA
X_train_ARIMA = X_train.loc[:, ['GOOGL', 'IBM', 'DEXJPUS', 'SP500', 'DJIA', 'VIXCLS']]
X_test_ARIMA = X_test.loc[:, ['GOOGL', 'IBM', 'DEXJPUS', 'SP500', 'DJIA', 'VIXCLS']]

tr_len = len(X_train_ARIMA)
te_len = len(X_test_ARIMA)
to_len = len(X)

model_ARIMA = ARIMA(endog=Y_train, exog=X_train_ARIMA, order=(1, 0, 0))
model_fit = model_ARIMA.fit()

error_training_ARIMA = mean_squared_error(Y_train, model_fit.fittedvalues)
predicted = model_fit.predict(start=int(tr_len - 1), end=int(to_len - 1), exog=X_test_ARIMA)[1:]

error_test_ARIMA = mean_squared_error(Y_test, predicted)

print(model_fit.summary())

# LSTM
seq_len = 2

Y_train_LSTM, Y_test_LSTM = np.array(Y_train)[seq_len-1:], np.array(Y_test)
X_train_LSTM = np.zeros((X_train.shape[0] + 1 - seq_len, seq_len, X_train.shape[1]))
X_test_LSTM = np.zeros((X_test.shape[0], seq_len, X_test.shape[1]))

for i in range(seq_len):
    X_train_LSTM[:, i, :] = np.array(X_train)[i:X_train.shape[0]+i+1-seq_len,:]
    X_test_LSTM[:, i, :] = np.array(X)[X_train.shape[0]+i-1:X.shape[0]+i+1-seq_len,:]

def create_LSTMmodel(learn_rate=0.01, momentum=0):
    model = Sequential()
    model.add(LSTM(50, input_shape=(X_train_LSTM.shape[1], X_train_LSTM.shape[2])))
    model.add(Dense(1))

    optimizer = SGD(lr=learn_rate, momentum=momentum)
    model.compile(loss='mse', optimizer='adam')

    return model

LSTMmodel = create_LSTMmodel(learn_rate=0.01, momentum=0)
LSTMmodel_fit = LSTMmodel.fit(X_train_LSTM, Y_train_LSTM, validation_data=(X_test_LSTM, Y_test_LSTM), epochs=512, batch_size=72, verbose=0, shuffle=True)

# pyplot.plot(LSTMmodel_fit.history['loss'], label='train')
# pyplot.plot(LSTMmodel_fit.history['val_loss'], '--', label='test')
# pyplot.legend()
# pyplot.show()

error_training_LSTM = mean_squared_error(Y_train_LSTM, LSTMmodel.predict(X_train_LSTM))
predicted = LSTMmodel.predict(X_test_LSTM)
error_test_LSTM = mean_squared_error(Y_test, predicted)
test_results.append(error_test_ARIMA)
test_results.append(error_test_LSTM)

train_results.append(error_training_ARIMA)
train_results.append(error_training_LSTM)

names.append("ARIMA")
names.append("LSTM")

fig = pyplot.figure()

width = 0.35
ind = np.arange(len(names))

# fig.suptitle('Algorithm Comparison')
# ax = fig.add_subplot(111)
# pyplot.bar(ind - width/2, train_results, width=width, label='Train Error')
# pyplot.bar(ind + width/2, test_results, width=width, label='Test Error')
# fig.set_size_inches(15, 8)
# pyplot.legend()
# ax.set_xticks(ind)
# ax.set_xticklabels(names)
# pyplot.show()

def evaluate_arima_model(arima_order):
    model_ARIMA = ARIMA(endog=Y_train, exog=X_train_ARIMA, order=arima_order)
    model_fit = model_ARIMA.fit()
    error = mean_squared_error(Y_train, model_fit.fittedvalues)
    return error

def evaluate_models(p_values, d_values, q_values):
    best_score, best_cfg = float('inf'), None

    for p in p_values:
        for d in d_values:
            for q in q_values:
                order = (p, d, q)
                try:
                    mse = evaluate_arima_model(order)
                    if mse < best_score:
                        best_score, best_cfg = mse, order
                    print(f'ARMIA{order}  MSE={mse}')
                except:
                    continue
    
    print(f'Best ARIMA{best_cfg}  MSE={best_score}')

p_values = [0, 1, 2]
d_values = range(0, 2)
q_values = range(0, 2)

evaluate_models(p_values, d_values, q_values)

model_ARIMA_tuned = ARIMA(endog=Y_train, exog=X_train_ARIMA, order=(2, 0, 1))
model_fit_tuned = model_ARIMA_tuned.fit()

predicted_tuned = model_fit_tuned.predict(start=tr_len -1, end=to_len -1, exog=X_test_ARIMA)[1:]

# predicted_tuned.index = Y_test.index
# pyplot.plot(np.exp(Y_test).cumprod(), 'r', label='actual')

# pyplot.plot(np.exp(predicted_tuned).cumprod(), 'b--', label='predicted')
# pyplot.legend()
# pyplot.rcParams['figure.figsize'] = (8, 5)
# pyplot.show()

from pickle import dump, Pickler

filename = 'finalized_model.pkl'

dump(model_ARIMA_tuned, open(filename, 'wb'))
