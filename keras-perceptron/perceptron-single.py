from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten #just two layers in this example

import wandb
from wandb.keras import WandbCallback

# logging code - specific to w&b site
run = wandb.init()
config = run.config

# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()

is_five_train = y_train == 5 #same size as y_train, contains 0 if not 5, 1 if 5
is_five_test = y_test == 5 #same as above
labels = ["Not Five", "Is Five"]

img_width = X_train.shape[1]
img_height = X_train.shape[2]

# create model
# only need input_shape in the first layer, keras will automatically apply shape to subsequent layers
model = Sequential() #instantiate model
model.add(Flatten(input_shape=(img_width, img_height))) #add initial layer Flatten, turns 2d matrix into 1d array of 784 length, has no learned parameters it's just performing a reshape, could be done in numpy
model.add(Dense(1)) #Dense because it is densely connected to previous layer, every element has a learned weight from the previous layer, the 1 means that we are outputing 1 number from this layer
model.compile(loss='mse', optimizer='adam',
              metrics=['accuracy'])

# Fit the model
# Note model trained with 3 epochs, data was passed through the model three times

model.fit(X_train, is_five_train, epochs=3, validation_data=(X_test, is_five_test),
          callbacks=[WandbCallback(data_type="image", labels=labels, save_model=False)])
model.save('perceptron.h5')

# See URL in output to wandb - use ctrl+insert to copy from the terminal
# Loss was measured as MSE