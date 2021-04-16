# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('XXXX.csv') #replace XXXX with the name of your file
X = dataset.iloc[:, start_column:end_column].values #separate data
y = dataset.iloc[:, label_column].values #separate label column

#encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[: ,2] = labelencoder_X_2.fit_transform(X[:,2])
onehotencoder = OneHotEncoder(categorical_features=[1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# part2 - building an ANN
# importing keras and other things needed

import keras
from keras.models import Sequential
from keras.layers import Dense

# initializing ANN
classifier = Sequential()

# adding input layer and first hidden layer
classifier.add(Dense(units = 6 , activation = 'relu' , kernel_initializer = 'uniform' , input_dim = XX  )) #replace XX with input dimension

#adding second hidden layer 
classifier.add(Dense(units = 6 , activation = 'relu' , kernel_initializer = 'uniform' ))

#adding output layer
classifier.add(Dense(units = 1, activation = 'sigmoid' , kernel_initializer = 'uniform'))

# compiling the ANN
classifier.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'] )

#fitting the ANN to the training set
classifier.fit(X_train , y_train, batch_size = 10 , epochs = 100) #change batch_size and epochs with your number

# part 3 - making the predictions and evaluating the model

# predicting the test results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test , y_pred)




