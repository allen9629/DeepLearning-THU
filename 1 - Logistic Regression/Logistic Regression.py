# -*- coding: utf-8 -*-
"""
Created on Tue Mar  3 10:30:05 2020

@author: allen
"""

import numpy as np
import mnist_data_loader
import matplotlib.pyplot as plt

mnist_dataset = mnist_data_loader.read_data_sets("./MNIST_data/", one_hot=False)
train_set = mnist_dataset.train
test_set = mnist_dataset.test
print('Training dataset size:' , train_set.num_examples)
print('Test dataset size:' , test_set.num_examples)
X=train_set.images
y=train_set.labels
y=np.reshape(y,(12049,1)) #reshape data
y=(y/3)-1 #let the data be only 0 and 1, make it easier to be classified
X_test=test_set.images
y_test=test_set.labels
y_test=np.reshape(y_test,(1968,1))
y_test=(y_test/3)-1
#print an example training image
example_id = 0
image = train_set.images[example_id] # shape = 784 (28*28)
label = train_set.labels[example_id] # shape = 1
print(label)
plt.imshow(np.reshape(image,[28,28]),cmap='gray')
plt.show()
class logistic_regression():    
    def __init__(self):        
        pass

    def sigmoid(self, x): #sigmoid function
        z = 1 / (1 + np.exp(-x))        
        return z    
        
    def initialize_params(self, dims): #define weights and bias
        weight = np.zeros((dims, 1))
        b = 0
        return weight, b    
    
    def costfunc(self, X, y, weight, b):
        m,n = X.shape
       
        a = self.sigmoid(np.dot(X, weight) + b)
        cost = -1 / m * np.sum(y * np.log(a) + (1 - y) * np.log(1 - a)) #Cross Entropy Error Function
        #gradient descent
        dw = np.dot(X.T, (a - y)) / m 
        db = np.sum(a - y) / m
#             
        return a, cost, dw, db    
        
    def training(self, X, y, alpha, epochs):
        weight, b = self.initialize_params(X.shape[1])
        costs = [] 
        trainaccuracy=[]
        for i in range(epochs):
            a, cost, dw, db = self.costfunc(X, y, weight, b)
            weight = weight - alpha * dw #weight update
            b = b - alpha * db     #bias update       
            if i % 100 == 0:
                costs.append(cost)   
                trainaccuracy.append(1-cost)
            if i % 100 == 0:
                print('Epoch %d cost %f' % (i, cost))
#            plt.plot(costs,'-.',color="red",label="Loss")
#            plt.plot(trainaccuracy,'-.',color="green",label="Accuracy")#plot loss and accuracy curve
#            plt.title('Loss and Accuracy Curve')
#            plt.ylabel('Loss/Accuracy')
#            plt.xlabel('Epoches (x100)')
#            plt.savefig('../lossacccurve.png')
        params = {
            'w': weight, 
            'b': b
        }
        grads = {            
            'dw': dw,            
            'db': db
        }        
        
        
        return cost, params, grads    
        
    def predict(self, X, params): #prediction
        y_prediction = self.sigmoid(np.dot(X, params['w']) + params['b'])        
        for i in range(len(y_prediction)):            
            if y_prediction[i] > 0.5: #Binary Classification by sigmoid function
                y_prediction[i] = 1
            else:
                y_prediction[i] = 0

        return y_prediction    
            
    def accuracy(self, y_test, y_pred): #check model accuracy
        matched = 0
        
        for i in range(len(y_test)):            
            for j in range(len(y_pred)):                
                if y_test[i] == y_pred[j] and i == j:
                    matched += 1
                   
        accuracy_score = matched / len(y_test)  
        return accuracy_score  
    
    
        
if __name__ == "__main__":
    model = logistic_regression()
    
    epochs=2000
    alpha= 0.06
    cost_list, params, grads = model.training(X, y, alpha, epochs)
    y_train_pred = model.predict(X, params)
    accuracy_score_train = model.accuracy(y, y_train_pred)
    print('Train accuracy is:', accuracy_score_train*100,"%")
    y_test_pred = model.predict(X_test, params)
    accuracy_score_test = model.accuracy(y_test, y_test_pred)
    print('Test accuracy is:', accuracy_score_test*100,"%")
    
    