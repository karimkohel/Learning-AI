#!/usr/bin/env python3

from keras import Sequential
from keras.layers import Dense, Activation

from keras.optimizers import Adam
from keras import backend as K
from keras.metrics import categorical_crossentropy

from data import train

samples, labels = train()


model = Sequential([
    Dense(16, input_shape=(1, ), activation='relu'),
    Dense(32, activation='relu'),
    Dense(2, activation='softmax'),
])

model.compile(Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(samples, labels, batch_size=10, shuffle=True, epochs=20, verbose=2)


print("DONE")