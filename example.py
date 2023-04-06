import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('track_sample1.dat', sep=' ')
#Index(['t[s]', 'ECG[a.u.]', 'Pletism[a.u.]'], dtype='object')
# 0: time, 1: ECG, 2: Plethy

t, E, P = data['t[s]'].to_numpy(), data['ECG[a.u.]'].to_numpy(), data['Pletism[a.u.]'].to_numpy()
E = (E - np.mean(E, axis=0)) / np.std(E, axis=0)
P = (P - np.mean(P, axis=0)) / np.std(P, axis=0)
data = np.column_stack((t, P))
#print(t, data.shape)

plt.plot(data[:, 0], data[:, 1], marker=',', label='Plethy')
plt.plot(t, E, marker=',', label='ECG')
plt.legend()
plt.show()


# create input/output sequences for LSTM
sequence_length = 50
X = []
y = []
for i in range(len(data) - sequence_length):
    X.append(data[i:i+sequence_length, :])
    y.append(data[i+sequence_length, 1])
X = np.array(X)
y = np.array(y)

# split data into training and testing sets
train_size = int(len(X) * 0.5)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

#print(X_train.shape)

# create and train LSTM model
model = Sequential()
model.add(LSTM(50, input_shape=(X.shape[1], X.shape[2])))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X_train, y_train, epochs=100, batch_size=64, )#validation_data=(X_test, y_test))


# predict test data using trained model
y_pred = model.predict(X_test)

# plot actual vs predicted signals
plt.figure(figsize=(10, 5))
plt.plot(y_test, label='Actual')
plt.plot(y_pred, label='Predicted')
plt.legend()
plt.title('Actual vs. Predicted Signal')
plt.xlabel('Time')
plt.ylabel('Signal')
plt.show()

