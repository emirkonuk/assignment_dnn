import numpy as np
from random import shuffle
#from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    numberOfSamples = X.shape[0]
    numberOfClasses = W.shape[1]
    
    for i in range(numberOfSamples):
        s = X[i].dot(W)
        s -= s.max()
        expS = np.exp(s)
        sumOfExpS = np.sum(expS)
        p = expS / sumOfExpS
        loss += -np.log(p[y[i]])
        
        for c in range(numberOfClasses):
            dW[:,c] += (p-(y[i]==c))[c]*X[i]
    
    loss /= numberOfSamples
    loss += reg*np.sum(W**2)
    dW /= numberOfSamples
    dW += 2*reg*W
    
    #pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg, X_val = None, y_val = None):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    
    numberOfSamples = X.shape[0]
    
    s = X.dot(W)
    s -= s.max(axis=1, keepdims=True)
    expS = np.exp(s)
    sumOfExpS = np.sum(expS, axis=1, keepdims=True)
    p = expS / sumOfExpS
    p_y = p[range(numberOfSamples), y]
    #divide by zero check
    p_y[p_y==0] = np.nextafter(0, 1)
    loss = np.sum(-np.log(p_y))
    loss /= numberOfSamples
    
    dS = p
    dS[range(numberOfSamples), y] -= 1
    dS /= numberOfSamples
    dW = X.T.dot(dS)
    
    dW += 2*reg*W
    loss += reg*np.sum(W**2)
    
    
    
    #pass
    #############################################################################
    #                          END OF YOUR CODE                                 #
    #############################################################################

    return loss, dW

