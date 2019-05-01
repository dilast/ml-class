# need to do pip install pillow to run

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
import numpy as np

model = InceptionV3(weights='imagenet')

img_path = 'nh.jpg'
img = image.load_img(img_path, target_size=(299, 299))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x) # normalizes and does some other stuff, this comes from the people who made the package
model.summary()
preds = model.predict(x)

print('Predicted:', decode_predictions(preds, top=3)[0]) # decodes the 1000+ sized perceptron that is output
#model.save('image.h5')
