'''
# CRISP-ML(Q)
1. Business & Data Understanding - 
Business Problem - It takes huge amount of manual effort in sorting the parcels
 to be sent to different regions.
Business Objective - Minimize Sorting Time
Business Constraints - Minimize the Cost

Success Criteria: 
    1. Business - Sorting time will be reduced by anywhere between 80% to 90%
    2. ML - Achieve an accuracy of > 93%
    3. Economic - Increase profits > 5%

Data Collection: 
    MNIST dataset handwritten Black & White images of digits with 28*28 pixels.
'''


# Import necessary libraries for MLP and reshaping the data structres
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

from sqlalchemy import create_engine


# Loading the data set using pandas as data frame format 
train_nn = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 6-d.Deep Learning & AI Part 3\Neural Network-ANN\train_sample.csv")
test_nn = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 6-d.Deep Learning & AI Part 3\Neural Network-ANN\test_sample.csv")

user = 'root'  # user name
pw = 'Seemscrazy1994#'  # password
db = 'digits_db'  # database name
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

train_nn.to_sql('train_nn', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
test_nn.to_sql('test_nn', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

sql1 = 'select * from train_nn;'
train = pd.read_sql_query(sql1, engine)

sql2 = 'select * from test_nn;'
test = pd.read_sql_query(sql2, engine)

# Separating the data set into 2 parts - all the inputs and label columns
# converting the integer type into float32 format 
x_train = train.iloc[:, 1:].values.astype("float32")
x_test = test.iloc[:, 1:].values.astype("float32")
y_train = train.label.values.astype("float32")
y_test = test.label.values.astype("float32")

# Normalizing the inputs to fall under 0-1 by 
# diving the entire data with 255 (max pixel value)
x_train = x_train/255
x_test = x_test/255

# # one hot encoding outputs for both train and test data sets 
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Storing the number of classes into the variable num_of_classes 
num_of_classes = y_test.shape[1]
x_train.shape
y_train.shape
x_test.shape
y_test.shape
 
# Creating a user defined function to return the model for which we are
# giving the input to train the ANN mode
# def design_mlp():

# Initializing the model 
model = Sequential()
model.add(Dense(150, input_dim = 784, activation = "relu"))
model.add(Dense(200, activation = "tanh"))
model.add(Dense(100, activation = "tanh"))
model.add(Dense(500, activation = "tanh"))
model.add(Dense(num_of_classes, activation = "softmax"))
model.compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = ["accuracy"])
#return model

model.summary()
# Building a model using train data set and validating on test data set

# fitting model on train data
model.fit(x = x_train, y = y_train, 
          batch_size = 1000, epochs = 20,
          verbose = 1, validation_data = (x_test, y_test))


# Prediction on test data
predict = model.predict(x_test, verbose = 1)

results = pd.DataFrame(np.argmax(predict, axis = 1) , columns = ['Label']) # Here we get the index of maximum value in the encoded vector
results

# Evaluating the model on test data  
eval_score_test = model.evaluate(x_test, y_test, verbose = 1)
# Accuracy on test data set
print ("Accuracy: %.2f%%" %(eval_score_test[1]*100)) 



# Accuracy score on train data 
eval_score_train = model.evaluate(x_train, y_train, verbose = 0)
print ("Accuracy: %.2f%%" %(eval_score_train[1]*100)) 


# Saving the model for Future Inferences
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("model.h5")

#################################################################################
#### Testing
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

loaded_model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

test = pd.read_csv(r"C:\Users\seema\OneDrive\Desktop\COURSE\360DigiTMG Course\Data Science_03052023_10AM(20032023)\Module 6-d.Deep Learning & AI Part 3\Neural Network-ANN\test.csv")

test_pred = test.values.astype('float32')/255

predictions = loaded_model.predict(test_pred)

result = pd.DataFrame(np.argmax(predictions, axis= 1), columns = ['Label'])
result
#################################################################################