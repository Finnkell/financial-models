import pandas as pd
import numpy as np

from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel

df = pd.read_csv('database/backtest_data.csv', sep=',')

cols = [col for col in df.columns]

y = pd.DataFrame(df['Qtd contratos'])
X = df.drop('Qtd contratos', axis=1)

scaler = StandardScaler()
scaler.fit(df)
df = pd.DataFrame(scaler.transform(df)).rename(columns={i:cols[i] for i in range(len(df.columns))})

scaler_y = StandardScaler()
scaler_y.fit(y)
y = pd.DataFrame(scaler_y.transform(y)).rename(columns={0: cols[-1]})

train_size = int(len(X) * (1 - 0.3))

X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

kernel = DotProduct() + WhiteKernel()

# model = SVR()
model = GaussianProcessRegressor(kernel=kernel)
model.fit(X_train, y_train)

predict = model.predict(X_test)

print(mean_squared_error(predict, y_test))
print(scaler_y.inverse_transform(predict))
