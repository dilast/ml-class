from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical

import wandb
from wandb.keras import WandbCallback

# logging code
run = wandb.init()
config = run.config

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
img_width = X_train.shape[1]
img_height = X_train.shape[2]

# one hot encode outputs
y_train = to_categorical(y_train) #to_categorical performs one hot encoding, shape of y_train will become 60_000 by 10
y_test = to_categorical(y_test)
labels = range(10)

num_classes = y_train.shape[1] #final dimension of y_train, which is 10

# create model
model = Sequential()
model.add(Flatten(input_shape=(img_width, img_height)))
model.add(Dense(num_classes, activation="sigmoid")) # Adds Dense layer with 10 perceptrons, outputs 10 numbers, will have 7850 weights, 784 per perceptron because of the one hot encoding plus 1 extra bias for each perceptron
model.compile(loss='mse', optimizer='adam',
              metrics=['accuracy'])

# Fit the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test),
          callbacks=[WandbCallback(data_type="image", labels=labels, save_model=False)])
model.save('model.h5')
