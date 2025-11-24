import tensorflow 
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.model_selection import train_test_split
import pickle
import numpy as np

data_dicc = pickle.load(open(r"TR\Pickles\data.pickle", "rb"))

data = np.asarray(data_dicc["data"])
labels = np.asarray(data_dicc["labels"])

x_train, x_test, original_y_train, original_y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

translation_dict = {
    "0" : [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
    "1" : [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
    "2" : [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
    "3" : [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
    "4" : [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
    "5" : [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
    "6" : [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
    "7" : [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
    "8" : [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.], 
    "9" : [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
}

y_train = np.empty((original_y_train.shape[0], 10), dtype=np.float32)
i = 0
for number in original_y_train:
    y_train[i] = translation_dict[str(number)]
    i += 1

y_test = np.empty((original_y_test.shape[0], 10), dtype=np.float32)
i = 0
for number in original_y_test:
    y_test[i] = translation_dict[str(number)]
    i += 1
