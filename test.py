#!/usr/bin/env python3

from keras import Sequential
from keras.layers import Dense, Activation



model = Sequential([
    Dense(16, input_shape=(1, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax'),
])


print("DONE")