import keras
from keras import layers, models, optimizers 
from keras import backend as k
from collections import namedtuple, deque
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.optimizers import Adam

# Data visualization 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import math
from numpy.random import choice
import random
from collections import deque

dataset = pd.read_csv('database\IBOV_D1.csv', sep='\t')

X = list(dataset["<CLOSE>"])
X = [float(x) for x in X]

validation_size = 0.2
train_size = int(len(X) * (1 - validation_size))
X_train, X_test = X[0:train_size], X[train_size:len(X)]

class Agent:
    def __init__(self, state_size, is_eval=False, model_name=''):
        self.state_size = state_size
        self.action_size = 3 # BUY, SELL, HOLD
        self.memory = deque(maxlen=1000)
        self.invetory = []
        self.model_name = model_name
        self.is_eval = is_eval

        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.model = load_model("models/" + model_name) if is_eval else self._model()

    def _model():
        model = Sequential()
        mode.add(Dense(units=64, input_dim=self.state_size, activation='relu'))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=8, activation='relu'))
        model.add(Dense(units=self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=0.001))

        return model

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)

        options = self.model.predict(state)

        return np.argmax(options[0])

    def exp_replay(self, batch_size):
        mini_batch = []
        memory_size = len(self.memory)

        for i in range(memory_size - batch_size + 1, memory_size):
            mini_batch.append(self.memory[i])

        for state, action, reward, next_state, done in mini_batch:
            target = reward

            if not done:
                target = reward + self.gamma*np.amax(self.model.predict(next_state)[0])

            target_f = self.model.predict(state)

            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=True)



        