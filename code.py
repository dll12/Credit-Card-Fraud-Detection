##Project: Classifying a bank transaction to be fraud or legitimate

#importing libraries
import math as m
import matplotlib.pyplot as mp
import numpy as np
import pandas as pa
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

#fetching data
ds=pa.read_csv('creditcard.csv')

#Spliting train and test data
train,test=train_test_split(ds,shuffle=True,train_size=0.8)

#Generating features
x_train=train.iloc[:,1:30].values
y_train=train.iloc[:,30].values
x_test=test.iloc[:,1:30].values
y_test=test.iloc[:,30].values


#using Logistic Regression
reg=LogisticRegression()
reg=reg.fit(x_train,y_train)
pre=reg.predict(x_test)

#using Random forest
reg=RandomForestClassifier(n_estimators=100)
reg.fit(x_train,y_train)
pre=reg.predict(x_test)

#using Arificial Neural Network
reg=MLPClassifier()
reg.fit(x_train,y_train)
pre=reg.predict(x_test)

#Accuracy
#Confusion Matrix
confusion_matrix()
confusion_matrix(y_test,pre)
accuracy_score(y_test,pre)








