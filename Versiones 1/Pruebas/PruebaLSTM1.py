import pickle
import numpy as np
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

data_dicc = pickle.load(open(r"TR\Pickles\data.pickle", "rb"))

data = np.asarray(data_dicc["data"])
labels = np.asarray(data_dicc["labels"])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)
x_train.reshape()
y_train
print(x_train.shape)
print(y_train.shape)

"""trainX = np.reshape(x_train, (x_train.shape[(0,0)], 1, x_train.shape[(0,0)]))
trainY= np.reshape(y_train, (y_train.shape[0], 1, y_train.shape[0]))"""


model = Sequential([tf.keras.Input(input_shape=(1600, 42))])
model.add(LSTM(4, input_shape=x_train))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
model.fit(x_train, y_train, epochs=100, batch_size=1, verbose=2)

f = open("modelLSTM.p", "wb")
pickle.dump({"model" : model}, f)
f.close()