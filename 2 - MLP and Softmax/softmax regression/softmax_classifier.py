import numpy as np


def softmax_classifier(W, input, label, lamda):
    """
      Softmax Classifier

      Inputs have dimension D, there are C classes, a minibatch have N examples.
      (In this homework, D = 784, C = 10)

      Inputs:
      - W: A numpy array of shape (D, C) containing weights.
      - input: A numpy array of shape (N, D) containing a minibatch of data.
      - label: A numpy array of shape (N, C) containing labels, label[i] is a
        one-hot vector, label[i][j]=1 means i-th example belong to j-th class.
      - lamda: regularization strength, which is a hyerparameter.

      Returns:
      - loss: a single float number represents the average loss over the minibatch.
      - gradient: shape (D, C), represents the gradient with respect to weights W.
      - prediction: shape (N, 1), prediction[i]=c means i-th example belong to c-th class.
    """

    ############################################################################
    # TODO: Put your code here
    def softmax(z):
       #define softmax function
       z_max=np.max(z)
       z= z-z_max
       num= np.exp(z)
       deno= 1.0/np.sum(num)
       zs= num.dot(deno)
       
       return zs
    
    m=input.shape[0]
    A=softmax(np.dot(input,W))
    loss=(-1/m)*np.sum(label*np.log(A))+0.5*lamda*np.sum(W*W)
    gradient= (-1/m)*np.dot(input.T,(label-A))+lamda*W
    prediction= np.argmax(input.dot(W),1)
      
    ###########################################################################
    return loss, gradient, prediction
