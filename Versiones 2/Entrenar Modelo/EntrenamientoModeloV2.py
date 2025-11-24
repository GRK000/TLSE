import pickle
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data_dicc = pickle.load(open(r"", "rb"))

data = np.asarray(data_dicc["data"])
labels = np.asarray(data_dicc["labels"])

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

print(x_train.shape)
print(y_train.shape + "\n")
dim_entrada = (x_train.shape[1],1)

model = Sequential()
model.add(LSTM(units=64, activation="relu", input_shape=dim_entrada))
model.add(Dense(units=1))
model.compile(optimizer="adam", loss="mse")

model.fit(x_train, y_train, epochs=20, batch_size=10)

y_predict = model.predict(x_test)
precision = accuracy_score(y_predict, y_test)

print("La precisión del modelo al ser entrenado ha sido del {}".format(precision))

f = open("modelV2.p", "wb")
pickle.dump({"model": model}, f)
f.close()