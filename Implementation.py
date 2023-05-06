from Instance import PerceptronClassifier # importing the model
from sklearn.metrics import confusion_matrix  # import needed for the confusion matrix
from sklearn.metrics import ConfusionMatrixDisplay # import needed to visualize the confusion matrix


import pandas as pd # importing pandas
import numpy as np # importing numpy
import matplotlib.pyplot as plt # importing matplotlib for the confusion matrix
from sklearn.model_selection import train_test_split # importing to split the data
from sklearn.preprocessing import LabelEncoder # importing to encode some data
encoder = LabelEncoder() # making the encoder a variable


df=pd.read_csv('ionosphere.data', delimiter=',', header=None) # import data
print(df.head(20)) # print - to do a view

dataset = df.values # making the df into a numpy array

X = dataset[:,0:34].astype(float) # making x the contiuous data
print(X) # see how it worked
Y = dataset[:,34]
print(Y) # see how it worked
encoder.fit(Y) # creates the mapping
Y = encoder.transform(Y) # implements the mapping
print(Y) # see how it worked

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123) # splitting the data and doing the 80/20 split
print(y_train) # seeing how it worked
y_train=y_train.astype(float) # making it a float
p = PerceptronClassifier(learning_rate=0.01, max_iters=1000) # calling the class
p.fit(X_train, y_train, learning_rate=0.01) # calling the fit function
predictions = p.predict(X_test) # putting the results from line 31 into the predict function


def accuracy(y_true, y_pred): #accuracy function
    accuracy1 = np.sum(y_true == y_pred) / len(y_true) # calculate accuracy
    return accuracy1 # return accuracy score

print("Perceptron classification accuracy", accuracy(y_test, predictions)) # finding the accuracy

from sklearn.metrics import classification_report
print(classification_report(y_test, predictions, labels=[1,0]))

print(confusion_matrix(y_test, predictions, labels=[1,0]))
cm_numbers=confusion_matrix(y_test, predictions, labels=[1,0])
display=ConfusionMatrixDisplay(confusion_matrix=cm_numbers, display_labels=[1,0])
display.plot()
plt.show()

print("")
print("")
print("")
print("TRAINING ERROR")
predictions2=p.predict(X_train) # finding the training error
print("Perceptron classification accuracy", accuracy(y_train, predictions2)) # finding the accuracy

print(classification_report(y_train, predictions2, labels=[1,0]))
print(confusion_matrix(y_train, predictions2, labels=[1,0]))
cm_numbers=confusion_matrix(y_train, predictions2, labels=[1,0])
display=ConfusionMatrixDisplay(confusion_matrix=cm_numbers, display_labels=[1,0])
display.plot()
plt.show()




