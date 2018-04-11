from __future__ import print_function

import numpy as np
from cs231n.classifiers.linear_svm import *
from cs231n.classifiers.softmax import *
#from past.builtins import xrange


class LinearClassifier(object):

    def __init__(self):
        self.W = None
        self.v = 0
        self.v_prev = 0

    def train(self, X, y, X_val=None, y_val=None, learning_rate=1e-3, reg=1e-5, num_iters=100,
            batch_size=100, momentum=0, decay=1, initialization='standard', verbose=False):
        """
        Train this linear classifier using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.
        - y: A numpy array of shape (N,) containing training labels; y[i] = c
          means that X[i] has label 0 <= c < C for C classes.
        - learning_rate: (float) learning rate for optimization.
        - reg: (float) regularization strength.
        - num_iters: (integer) number of steps to take when optimizing
        - batch_size: (integer) number of training examples to use at each step.
        - verbose: (boolean) If true, print progress during optimization.

        Outputs:
        A list containing the value of the loss function at each training iteration.
        """
        num_train, dim = X.shape
        num_classes = np.max(y) + 1 # assume y takes values 0...K-1 where K is number of classes
        if self.W is None:
            if initialization == 'standard':
                # lazily initialize W
                self.W = 0.01 * np.random.randn(dim, num_classes)
            elif initialization == 'xavier':
                # or initialize with Xavier
                std = np.sqrt(1/dim)
                self.W = np.random.normal( 0, std, (dim, num_classes))           

        
        #use this if you want to use the "epochs" convention instead of "iterations"  
        #idxs = np.random.permutation(np.arange(num_train))
        #batches = [idxs[batch_id*batch_size:(batch_id+1)*batch_size] \
        #           for batch_id in range(num_train//batch_size+1)]

        # Run stochastic gradient descent to optimize W
        loss_history = []
        val_loss_history = []
        train_loss_history = []
        
        # define a counter for epochs
        it_epoch = -1
        iters_in_an_epoch = num_train//batch_size
        
        initial_learning_rate = learning_rate
        for it in range(num_iters):
            #X_batch = None
            #y_batch = None

            #########################################################################
            # TODO:                                                                 #
            # Sample batch_size elements from the training data and their           #
            # corresponding labels to use in this round of gradient descent.        #
            # Store the data in X_batch and their corresponding labels in           #
            # y_batch; after sampling X_batch should have shape (dim, batch_size)   #
            # and y_batch should have shape (batch_size,)                           #
            #                                                                       #
            # Hint: Use np.random.choice to generate indices. Sampling with         #
            # replacement is faster than sampling without replacement.              #
            #########################################################################
            idx = np.random.choice(num_train, batch_size)
            X_batch = X[idx]
            y_batch = y[idx]
            #pass
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################
            # nesterov
            self.W += momentum*self.v 
            
            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)
            
            self.v_prev = self.v
            self.v = momentum*self.v - learning_rate*grad
            self.W += -momentum*self.v_prev + (1+momentum)*self.v
            
            # learning rate decay
            learning_rate = initial_learning_rate * decay**(it//iters_in_an_epoch)
            
            # check if we are in a new epoch
            # and calculate train/validation losses for whole data sets
            if it_epoch < (it//iters_in_an_epoch):   
                val_numberOfSamples = X_val.shape[0]
                val_s = X_val.dot(self.W)
                val_s -= val_s.max(axis=1, keepdims=True)
                val_expS = np.exp(val_s)
                val_sumOfExpS = np.sum(val_expS, axis=1, keepdims=True)
                val_p = val_expS / val_sumOfExpS
                val_p_y = val_p[range(val_numberOfSamples), y_val]
                #divide by zero check
                val_p_y[val_p_y==0] = np.nextafter(0, 1)
                val_loss = np.sum(-np.log(val_p_y))
                val_loss /= val_numberOfSamples
                val_loss += reg*np.sum(self.W**2)
                val_loss_history.append(val_loss)
                
                numberOfSamples = X.shape[0]    
                s = X.dot(self.W)
                s -= s.max(axis=1, keepdims=True)
                expS = np.exp(s)
                sumOfExpS = np.sum(expS, axis=1, keepdims=True)
                p = expS / sumOfExpS
                p_y = p[range(numberOfSamples), y]
                #divide by zero check
                p_y[p_y==0] = np.nextafter(0, 1)
                train_loss = np.sum(-np.log(p_y))
                train_loss /= numberOfSamples
                train_loss += reg*np.sum(self.W**2)
                train_loss_history.append(train_loss)
                
                it_epoch = it//iters_in_an_epoch
                #print('Finished %i epochs' % (it_epoch+1))
            

            # perform parameter update
            #########################################################################
            # TODO:                                                                 #
            # Update the weights using the gradient and the learning rate.          #
            #########################################################################
            # vanilla update
            #self.W += -learning_rate*grad
            
            
            
            #pass
            #########################################################################
            #                       END OF YOUR CODE                                #
            #########################################################################

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history, val_loss_history, train_loss_history

    def predict(self, X):
        """
        Use the trained weights of this linear classifier to predict labels for
        data points.

        Inputs:
        - X: A numpy array of shape (N, D) containing training data; there are N
          training samples each of dimension D.

        Returns:
        - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
          array of length N, and each element is an integer giving the predicted
          class.
        """
        y_pred = np.zeros(X.shape[0])
        ###########################################################################
        # TODO:                                                                   #
        # Implement this method. Store the predicted labels in y_pred.            #
        ###########################################################################
        s = X.dot(self.W)
        y_pred = np.argmax(s, axis=1)
        
        #pass
        ###########################################################################
        #                           END OF YOUR CODE                              #
        ###########################################################################
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        """
        Compute the loss function and its derivative. 
        Subclasses will override this.

        Inputs:
        - X_batch: A numpy array of shape (N, D) containing a minibatch of N
          data points; each point has dimension D.
        - y_batch: A numpy array of shape (N,) containing labels for the minibatch.
        - reg: (float) regularization strength.

        Returns: A tuple containing:
        - loss as a single float
        - gradient with respect to self.W; an array of the same shape as W
        """
        pass


class LinearSVM(LinearClassifier):
    """ A subclass that uses the Multiclass SVM loss function """

    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)

