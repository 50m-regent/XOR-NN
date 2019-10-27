from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD

from numpy import array

teacher_input = array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

teacher_output = array([0, 1, 1, 0])

model = Sequential()
model.add(Dense(2, activation='tanh'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1))
history = model.fit(teacher_input, teacher_output, batch_size=1, epochs=5000)

print(model.predict(teacher_input))
