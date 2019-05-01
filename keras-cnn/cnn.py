from keras.datasets import mnist
from keras.models import Sequential
#imports different types of layers
from keras.layers import Conv2D, MaxPooling2D, Dropout, Dense, Flatten
from keras.utils import np_utils
from wandb.keras import WandbCallback
import wandb
import os

run = wandb.init()
#set hyperparameters
config = run.config
config.first_layer_convs = 32
config.first_layer_conv_width = 3 #kernel width
config.first_layer_conv_height = 3 #kernel height
config.dropout = 0.2
config.dense_layer_size = 128
config.img_width = 28
config.img_height = 28
config.epochs = 10

(X_train, y_train), (X_test, y_test) = mnist.load_data()

# normalize data
X_train = X_train.astype('float32')
X_train /= 255.
X_test = X_test.astype('float32')
X_test /= 255.

# reshape input data
X_train = X_train.reshape(
    X_train.shape[0], config.img_width, config.img_height, 1) # 60_000 x 28 x 28 x 1
#convolution layers always expect 3D input, just wrapping array in another array to get extra channel dimension
#see common error in the ML Class Apr 4 note

X_test = X_test.reshape(
    X_test.shape[0], config.img_width, config.img_height, 1)

# one hot encode outputs
# creates 10-length vector for digits 0-9
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
labels = range(10)

# build model
model = Sequential()

# second argument is convolution size, here 3x3
model.add(Conv2D(32,
                 (config.first_layer_conv_width, config.first_layer_conv_height),
                 input_shape=(28, 28, 1),
                 activation='relu'))
# shape of data coming out of this convolution layer is 26x26x32 - performing 32 convolutions, so outputing 32 feature maps - and 26 because there is no padding, which is the default, the convolution loses dimensions based on the kernel size, we lose pixels on the height and width. We can fix by putting 0s around the input map

# also, by default, convolution has a stride of 1, but you can change this to be larger if your data is very large - this will make the feature map smaller, similar to maxpooling, because it will skip over pixels

model.add(MaxPooling2D(pool_size=(2, 2)))
# shape of data coming out is 13x13x32, because it resizes the output of the previous convolution layer

# add some more stuff
model.add(Dropout(0.4))
model.add(Conv2D(64, (3,3), activation="relu"))
model.add(MaxPooling2D())

model.add(Flatten())

model.add(Dense(config.dense_layer_size, activation='relu'))

# add some dropout - can put it anywhere, but generally wouldn't put it after last layer
model.add(Dropout(0.4))

model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])


model.fit(X_train, y_train, validation_data=(X_test, y_test),
          epochs=config.epochs,
          callbacks=[WandbCallback(data_type="image", save_model=False)])
