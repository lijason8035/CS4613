#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 21:23:06 2018

@author: JasonLi
"""

from matplotlib import pyplot as plt
import numpy as np
from numpy import array
import struct
import random
import pandas as pd

def byteToPixel(file, width,length):
    # Create a string for the struct function to decode the bytes to interger
    stringcode = '>'+'B' * len(file)
    data = array(struct.unpack(stringcode, file))
    # Reshape the data to number of images times number of pixel in each images
    data = data.reshape(int(len(file)/(width*length)),width*length,1)/255
    return data

def load_data(filename, labelname, width, length):
    #Loading raw files
    with open(filename, 'rb') as f:
        bytefile = f.read()
    f.close()
    pixelfile = byteToPixel(bytefile, width,length)
    
    #Loading label files
    file = open(labelname,"r")
    labelfile = []
    for line in file:
        current = line.split()
        results = list(map(int, current))
        results = array(list(map(lambda x:[x], results)))
        labelfile.append(results)
    
    #Use List to combine image data file with its label file
    data = list(zip(pixelfile, labelfile))
    return data

def confusionMatrix(result,outterlayer):
    #Create a matrix of zeros
    Matrix = np.zeros((outterlayer,outterlayer))
    #Count up by one depends on the true label vs the predict label
    for (x,y) in result:
        Matrix[np.argmax(y)][np.argmax(x)] += 1
    Matrix = pd.DataFrame(Matrix)
    return Matrix

def sigmoid(z):
    # The sigmoid function
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    # Derivative of the sigmoid function
    return sigmoid(z)*(1-sigmoid(z))

class Network(object):

    def __init__(self, sizes):
        self.num_hlayers = len(sizes)-2
        self.sizes = sizes
        self.biases = []
        self.weights = []
        
    def create_wb(self, weights,biases):
        self.biases = biases
        self.weights = weights
    
    def return_wb(self):
        return self.weights, self.biases

    def feedforward(self, activation):
        # Calculate the activation for every layer but it will only output final layer
        for b, w in zip(self.biases, self.weights):
            activation = sigmoid(np.dot(w, activation)+b)
        return activation
    
    def pick_mini_batches(self,training_data,mini_batch_size):
        #Instead of the trainning the whole data set
        #Randomly shuffle the data and then pick out x number of size of images for 
        #each batch until we exhuast the entire dataset
        n = len(training_data)
        random.shuffle(training_data)
        mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
        return mini_batches
        
    def training_network(self, training_data,mini_batch_size,learning_rate):
            mini_batches = self.pick_mini_batches(training_data,mini_batch_size)
            for mini_batch in mini_batches:
                #For each mini batches, we would record the total change of weights and biases in that batch
                #Depending on the learning rate
                self.changed_mini_batch(mini_batch, learning_rate)
    
    def changed_mini_batch(self, mini_batch, learning_rate):
        #Create a place holder for the total change of weights and biases
        sum_of_b = [np.zeros(b.shape) for b in self.biases]
        sum_of_w = [np.zeros(w.shape) for w in self.weights]
        
        #In each batches, we have the initial layer of activation and the final activation(label)
        for activation, label in mini_batch:
            #Use back propagation to calculate a SINGLE image of changes in weights and biases
            delta_one_b, delta_one_w = self.backward_prop(activation, label, learning_rate)
            
            #Sum of the changes
            sum_of_b = [sb+dob for sb, dob in zip(sum_of_b, delta_one_b)]
            sum_of_w = [sw+dow for sw, dow in zip(sum_of_w, delta_one_w)]

        #Equations for calculating the total weights and biases
        new_weights = []
        new_biases = []
        for w, sw in zip(self.weights, sum_of_w):
            new_weights.append(w-(1/len(mini_batch))*sw)
        for b, sb in zip(self.biases, sum_of_b):
            new_biases.append(b-(1/len(mini_batch))*sb)
        self.weights = new_weights
        self.biases =new_biases
    
    def backward_prop(self,x, y, learning_rate):
        #Create a place holder for the SINGLE change of weights and biases
        change_b = [np.zeros(b.shape) for b in self.biases]
        change_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        # list to store all the activations, layer by layer
        activation_collection = [activation]
        # list to store all the z vectors, layer by layer (Before passing through sigmoid function)
        zlayers = []
        
        # feedforward
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zlayers.append(z)
            activation = sigmoid(z)
            activation_collection.append(activation)
            
        #Calculate backward from label y 
        #Calculate the changes of biases and weights in the last layer depending on the changes of the total cost  
        delta = self.cost_derivative(activation_collection[-1], y) * sigmoid_prime(zlayers[-1])
        change_b[-1] = delta
        change_w[-1] = np.dot(delta, activation_collection[-2].transpose())
        
        #Calculate the changes of biases and weights from the second to last layer forward.
        for layer in range(2, self.num_hlayers+2):
            z = zlayers[-layer]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-layer+1].transpose(), delta) * sp * learning_rate
            change_b[-layer] = delta
            change_w[-layer] = np.dot(delta, activation_collection[-layer-1].transpose())
        return change_b, change_w
    
    def evaluate(self, test_data):
        #Send Test Data to the function feed forward 
        test_results = [(self.feedforward(x), y)for (x, y) in test_data]
        #Result is number of correct predicted labels
        result = 0
        for (x, y) in test_results:
            if np.argmax(x) == np.argmax(y):
                result += 1
        mse = 0
        #Calculate the root mean squre error
        for item in test_results:
            temp = sum((item[0] - item[1])**2)/len(item[1])
            mse = temp+mse
        mse = mse/len(test_results)
        
        return result, mse, test_results
        
    #Derivative of the cost function
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)
    
    def evaluate_epochs(self,training_data,mini_batch_size,learning_rate, test_data=None):
        
        if test_data is not None:
            n_test = len(test_data)
        #Place holder for calculating the differences in root mean square error
        mse_diff = 0
        a_mse = 0
        b_mse = 0
        #Keep track of number of epochs
        j = 0
        #Keep track of number of correct predicted label, root mean square error and differences of the rmse
        result_epochs = []
        result_mse = []
        result_mse_different = []
        while True:
            self.training_network(training_data,mini_batch_size,learning_rate)
            if test_data is not None:
                result ,mse,test_results = self.evaluate(test_data)
                result_epochs.append(result)
                result_mse.append(mse)
                a_mse = mse
                #If the difference between root mean square barely changes (threshold: 0.0001), then we stop training
                if mse_diff <0.0001 and mse_diff != 0:
                    print ("Complete!")
                    return test_results, result_epochs, result_mse, result_mse_different
                else:
                    #Printing out the result of the each epoch run
                    mse_diff = abs(a_mse - b_mse)
                    if mse_diff == a_mse:
                        mse_diff = None
                    result_mse_different.append(mse_diff)
                    b_mse = a_mse
                    print ("Epoch " + str(j)+ ": "+ str(result)+' / ' + str(n_test) + '   ' + str(mse)+'   ' \
                           + str(mse_diff))
                    if mse_diff is None:
                        mse_diff = 2
            else:
                return test_results, result_epochs, result_mse, result_mse_different
                print ("Epoch" + str(j) +  "complete")

            j+=1

def main():
    train_image = 'train_images.raw'
    test_image = 'test_images.raw'
    train_label = "train_labels.txt"
    test_label = "test_labels.txt"
    contents_train = load_data(train_image,train_label,28,28)
    contents_test = load_data(test_image,test_label,28,28)
    
    sizes = [784,50,5] #Change number of neurons here 
    net = Network(sizes)
    w = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
    b = [np.ones((y, 1)) for y in sizes[1:]]
    net.create_wb(w,b)
    print ("Epochs                 MSE            MSE Difference")
    result, result_epochs, result_mse, result_mse_different= net.evaluate_epochs\
    (contents_train,100,0.1, test_data=contents_test)
    
    print ("")
    print ("Confusion Matrix")
    print ("")
    matrix = confusionMatrix(result,5)
    print (matrix)
    
    result_accuracy = [x / len(contents_test)*100 for x in result_epochs]
    print ("")
    print ("Accuracy vs Number of Epochs")
    plt.figure(0)
    plt.plot(result_accuracy)
    plt.xlabel('Number of Epochs', fontsize=20)
    plt.ylabel('Accuracy (%)', fontsize=20)
    plt.show()
  
main()