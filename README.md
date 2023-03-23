# Instructions for Perceptron Algorithm Assignment

Course: 5104 - Deep Learning
Spring 2023

---
## Problem Background

#### Binary classification:
Binary classification refers to the task of "classifying" the labels of two categories of data _(e.g. classifying whether a bank loan will default or not)._ Our task as Data Scientists is to identify how to clearly separate the two classes of data using a straight line or hyperplane.

The image below shows an algorithm that can be used for binary classification tasks called the __perceptron__. You will notice that the perceptron looks very similar in operation to the tensor operations in a `Dense` layer of a neural network. One difference is the perceptron only outputs values of `[0, 1]`. A simple way to understand perceptron is to envision a single layer neural network that uses a binary step activation function. The binary step output tells us the cateorgical labels using 0 (negative cases) or 1 (positive cases).
![img2](static/img/perceptron.webp)

#### What is the Perceptron?
In the 1940s and 50s a group of scientists, including Warren McCulloch, Walter Pitts and Frank Rosenblatt published a series of papers describing algorithms based on the brain. They observed that the brain is a network, made up of billions of nerve cell connectors called neurons. 

Neurons are comprised of branch-like receptors called dendrites, that receive electrical impulses from other upstream neural cells, and a long, thin projection called an axon that sends signals downstream to other neurons. Once a neurons’ dendrites receive an impulse, if a certain threshold is reached, the neuron fires and it sends a signal to other neurons through its axon. Since neurons are connected in a large network, when an impulse is received by one neuron, it can set off a wide range of electrical activity within the brain, creating a chain reaction.

![img](static/img/neuron.png)

McCulloch, Pitts and Rosenblatt realized that a simplified version of this neurological process could be used to implement a self-learning computer function. They believed that the brain is able to update the strength of its neural connections over time as new experiences are learned. Based on this approach, they developed the Perceptron Learning Algorithm.

#### So what makes the perceptron algorithm useful?

The perceptron allows us to determine the best hyperplane (line) that separates two classes of data. This allows us to say data on one side of the hyperplane belongs to the positive labels (1) or the negative labels (0). The labels may represent a category or event we'd like to predict, such as:
-  who will drive to hammonton for class or not?
-  who will default on their mortgages in the next 6 months?
-  does an image contain a cat or not?
-  who will pass the first take home exam in deep learning?
-  Will it rain tomorrow or not?
![img](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/images/perceptron/perceptron_img1.png)


### Assumptions (before you start coding):
Lets review the assumptions of the orginal perceptron.
- The perceptron output must be binary `0` for the negative label or `1` for the positive label
- The input data must be linearly separable, or the algorithm will never converge. 
    > The convergance assumption is what led to disinterest in the original design of the perceptron. However, other variants (like the one we will make), relaxes this contraint to allow the algorithm more flexibility. 


## Algorithm

Below is the algorithm of the original perceptron that you will need to implement in python code. Please read the Coding Objects for complete details on what you must include in your submission of the perceptron. 
![img](static/img/algorithm.webp)

## Weight Update
The original algorithm for the perceptron (like the image above), did not actually utilize a learning rate as part of the weight updates. Instead, the algorithm repeated until it converged. This becomes problematic for data that may be mislabeled or is not 100% separated. In our version of the algorithm we will alter the weight update as follows:

1. __Calculating the error of the prediction__: the predicted output (1 or 0) is compared to the actual outcome in the training data. `error = yhat - y`
2.  __Adjust the error based on the learning rate__: Next, the error is multiplied by a learning rate, typically a number between 0 and 1. The learning rate is used to  determine the extent in which new information is allowed to impact the existing weight values.  `adj_error = error x learning_rate`

    > A smaller learning rate means smaller impact on the change in yhat, or smaller impact from the new information it learns.
    > Using a smaller learning rate has its drawbacks, for example, it can significantly slow down the time it takes to train your algorithm. 

3. __Update the weights__: The weight update is a step that "adjusts" the current weight's values  in the direction of the error. This update allows the perceptron to  make more accurate predictions.
  - `W = W + adj_error * xi`
  - `b = b + adj_error`
   > This is how the perceptron learns... It uses the weights to make a prediction, checks the outcome against the actual result, then adjusts its weights accordingly. It also means that the more training data the perceptron is fed, the more this training loop can improve the perceptron's accuracy.

---

## Coding Objectives:


1. __(30% of assignment)__ Write the perceptron algorithm <u>from scratch using python.</u>
   - You may use `numpy`,`pandas` or any python standard libraries for performing tensor operations, but you cannot use any packages that provide a direct implmentation of the perceptron for you.
   - You must comment every line of code you write and explain what it does. __Failure to comment EVERY LINE of your code will cost you all points for this section of the assignment, No Exceptions__.
   - Your Algorithm must be a Python Class that implements the following:
       - the python class must be  called `PerceptronClassifier`
       - Include an `__init__` constructor that initializes any attributes you need to run your training procedure. You can optionally initialize the weights `W` and `b` randomly or with zeros when the class is instantiated as long as the attributes are updated with random or zero starting values when the training procedure begins).
       - A __method__ called `fit()` must be used to execute the training procedure.
         - The fit method must accept `X : np.array`, `y : np.array`, `learning_rate : float`, as default arguments
         - <u>Optionally</u>, you may include a `max_iter : int` argument that can end training before convergence. This may be a helpful feature to reduce overfitting.
   
       - A __method__ called `predict()` which uses the trained parameters of the classifier to make predictions.
         - The predict __method__ should implements the following `yhat = binary_step_activation(dot(X,W)+b)`
      - A python __function__ called `binary_step_activation` which computes the binary step activation function (output is 1 or 0). See `np.heaviside` or you can implement your own from scratch.
        ![img](static/img/binary_step.png)
  

2. __(30% of assignment)__ You must write a python script `.py` or jupyter notebook `.ipynb` that uses your perceptron and the ionosphere dataset found [here](https://archive.ics.uci.edu/ml/datasets/Ionosphere).
   - You may use the `scikit-learn` library for evaluating, preprocessing, or feature engineering etc...
   - You must divide the dataset into train and test sets using a 80/20 split.
   - You must decide which features to include or exclude from when training your model.
   - You may do any feature engineering or data transformations you'd like that might improve the performance of your model. 
   - Complete your training procedure and use the trained model to make predictions using the test set. 
   - Create and save a line plot that shows your training error at each iteration of training. Make note of any observations in your report below.


3. __(40% of assignment)__ Evaluate the performance of your model on the test set and answer the following in a separate markdown file named `report.md`:
   -   What is the difference between your training error and test error? Is your model overfit or underfit? How does this impact the usefulness of your model?
    - Make a confusion matrix showing the results of your model on both training and test sets. Incude these in your markdown with an explaination of the results. To include an image in your markdown use something like this:
        ```markdown
        ![img](path/to/my/image.png)
        ```
    - Calcuate the following metrics and explain each one of them and their relevance to your model in the markdown report. 
      - Percision
      - Recall
      - F1-Measure
      - Accuracy
