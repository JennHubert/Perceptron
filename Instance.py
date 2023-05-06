#Jennifer Hubert # my name
import pandas as pd # import pandas 
import numpy as np # import numpy
import matplotlib.pyplot as plt # importing matplotlib for the line plot !



class PerceptronClassifier(): # creating my class !
    
    def __init__(self, learning_rate=0.01, max_iters=1000): # constructor to instantiate things !
        self.W = None # instantiating the weights
        self.b = None # instantiating the bias
        self.binary_step_activation = self.binary_step_activation # instantiating the function
        self.learning_rate=learning_rate # instantiating the learning rate
        self.max_iters=max_iters # instantiating the max iterations
        self.train_errors = [] # creating an empty list to contain the training errors for the line plot later !
        
    def fit(self, X : np.array, y : np.array, learning_rate : float): # fit function - where the fun is !
        n_samples, n_features = X.shape # defining the samples and features

        self.W = np.zeros(n_features) # creating the weights - an array of zeros ! (the number of features to be its shape)
        self.b = 0 # starting the bias out at ZERO
        
        y_ = np.where(y>0,1,0) # for y_, if the value is greater than zero, 1 will be inserted, if not - zero will be inserted
        
        for _ in range(self.max_iters): # for each iteration
            errors = 0 # defining errors for the line plot later
            for idx, xi in enumerate(X): # iterating
                linear_output = np.dot(xi, self.W) + self.b # finding the output
                y_predicted= self.binary_step_activation(linear_output) # finding the predicted values
                
                error = y_[idx] - y_predicted # finding the error
                update = self.learning_rate * error # determining how the weights and bias needs to be changed
                self.W = self.W + update * xi # updating the weights
                self.b = self.b + update # updating the bias
                
                errors += int(update != 0)  # counting the errors for this iteration
            self.train_errors.append(errors / n_samples)  # computing the training error and adding it to the list
            
        # creates a line plot of the training error values
        plt.plot(self.train_errors) # instantiates
        plt.xlabel('Iteration') # labels x axis
        plt.ylabel('Training Error') # labels y axis
        plt.title('Perceptron Training Error') # makes the title
        plt.show # shows plot

    def predict(self, X): # makes the predictions based on the fit function
        yhat=self.binary_step_activation(np.dot(X, self.W) + self.b) # calls the activation function with the prediction values
        return yhat # returns predictions
    

    def binary_step_activation(self, linear_output): # activation function - I was getting an error when using a more basic version of this
        binary_output = np.zeros(linear_output.shape)  # makes a new array with the shape of the existing one
        binary_output[linear_output >= 0] = 1 # makes the necessary change to binary values
        return binary_output # returns the output - the final predictions !
    # the above function was the most frustraing part of this

     

