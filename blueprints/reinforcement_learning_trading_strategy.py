import tensorflow.keras
from tensorflow.keras import layers, models, optimizers 
from tensorflow.keras import backend as k
from collections import namedtuple, deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.activations import sigmoid

# Data visualization 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import math
from numpy.random import choice
import random
from collections import deque

dataset = pd.read_csv('database\WIN$N_M1.csv', sep=',')

X = list(dataset["Close"])
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

    def _model(self):
        model = Sequential()
        model.add(Dense(units=64, input_dim=self.state_size, activation='relu'))
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

            self.model.fit(state, target_f, epochs=1, verbose=False)


def get_state(data, t, n):
    d = t - n + 1

    block = data[d:t + 1] if d >= 0 else -d * [data[0]] + data[0:t + 1]
    res = []

    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))

    return np.array([res])

def plot_behavior(data_input, states_buy, states_sell, profit):
    fig = plt.figure(figsize=(15, 5))
    plt.plot(data_input, color='r', lw=2)
    plt.plot(data_input, '^', markersize=10, color='purple', label='Buying signal', markevery=states_buy)
    plt.plot(data_input, 'v', markersize=10, color='yellow', label='Selling signal', markevery=states_sell)
    plt.title(f'Total gain: {profit}')
    plt.legend()
    plt.show()

data = X

window_size = 1
agent = Agent(window_size)
data_size = len(data) - 1
batch_size = 10
states_sell = []
states_buy = []
episode_count = 3

for e in range(episode_count + 1):
    print(f'Episode: {e}/{episode_count}')

    state = get_state(data, 0, window_size + 1)

    total_profit = 0
    agent.invetory = []

    for t in range(data_size):
        action = agent.act(state)

        next_state = get_state(data, t + 1, window_size + 1)
        reward = 0

        if action == 1:
            agent.invetory.append(data[t])
            states_buy.append(t)

            print(f'Buy: {data[t]}')

        elif action == 2 and len(agent.invetory) > 0:
            bought_price = agent.invetory.pop(0)

            reward = max(data[t] - bought_price, 0)
            total_profit += data[t] - bought_price
            states_sell.append(t)

            print(f'Buy at: {bought_price} | Sell: {data[t]} | Profit: {data[t] - bought_price}')

        done = True if t == data_size - 1 else False

        next_state = get_state(data, t + 1, window_size + 1)

        agent.memory.append((state, action, reward, next_state, done))
        state = next_state

        if done:
            plot_behavior(data, states_buy. states_sell, profit)
            print(f'+--------------------------+\n| Total Profit : ${total_profit}\n+--------------------------+')

        if len(agent.memory) > batch_size:
            agent.exp_replay(batch_size)

    if e % 10 == 0:
        agent.model.save("models/model_rlts_ep" + str(e))
