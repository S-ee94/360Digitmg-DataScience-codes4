import tensorflow as tf


tf.test.is_gpu_available(
    cuda_only=False, min_cuda_compute_capability=None
)


import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from sqlalchemy import create_engine

# Design the Network
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.summary()

# Num of Parameters
# [(w*h*d)+1]*k 
# w = width; h = height; d = filters from previous layer
# k = current layer filters

model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()


# Load the MNIST dataset
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# reshaping test_images to 2d 
arr_reshaped = test_images.reshape(test_images.shape[0], -1)

# converting reshaped array to dataframe
df = pd.DataFrame(arr_reshaped)  

# selecting 100 rows randomly
df1 = df.sample(n = 100)

# Saving small amount of data for testing
df1.to_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 6-f.Deep Learning & AI CNN\CNN\test_cnn.csv", index=False)

user = 'root'  # user name
pw = 'Seemscrazy1994#'  # password
db = 'cnn_db'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

df1.to_sql('cnn_test', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
test_acc


model_json = model.to_json()

with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")




#######################
# Testing on new data

from tensorflow.keras.models import model_from_json

import pandas as pd

# from keras import model_from_json 

# opening and store file in a variable

json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

# use Keras model_from_json to make a loaded model
loaded_model = model_from_json(loaded_model_json)

# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded Model from disk")

# compile and evaluate loaded model
# loaded_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

test1 = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 6-f.Deep Learning & AI CNN\CNN\test_cnn.csv")


arr_img = test1.to_numpy()

test_pred = arr_img.reshape((len(arr_img), 28, 28, 1))

test_pred = test_pred.astype('float32') / 255

predictions = pd.DataFrame(loaded_model.predict(test_pred))

