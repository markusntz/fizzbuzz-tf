# fizzbuzz tensor style

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout

NUM_DIGITS = 10

# represent each digit by an array of its binary digits.
def binary_encode(i, num_digits):
    return np.array([i >> d & 1 for d in range(num_digits)])

# one hot encode results: [number, "fizz", "buzz", "fizzbuzz"]
def fizz_buzz_encode(i):
    if i % 15 == 0: return np.array([0, 0, 0, 1])
    elif i % 5  == 0: return np.array([0, 0, 1, 0])
    elif i % 3  == 0: return np.array([0, 1, 0, 0])
    else: return np.array([1, 0, 0, 0])

# see one hot encoder in action
[fizz_buzz_encode(x) for x in [10, 15, 45, 34]]

# create training data
x_train = np.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
y_train = np.array([fizz_buzz_encode(i) for i in range(101, 2 ** NUM_DIGITS)])

model = Sequential()
model.add(Dense(64, activation='relu', input_dim=NUM_DIGITS))
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=500, batch_size=64, verbose = 2)

numbers = np.arange(1, 101)
x_test = np.transpose(binary_encode(numbers, NUM_DIGITS))

classes = model.predict(x_test, batch_size=128)
preds = np.argmax(classes, axis=1)

def fizz_buzz(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

out = np.vectorize(fizz_buzz)(numbers, preds)
out
# mhm definitely needs more layers...
