'''
Auxiliary file containing defintions of functions.
'''
#Import libraries
import numpy as np
from numba import cuda

#Import auxiliary file
import cuda_backend

class Function():
    '''
    Class of a function that can be used as an activation in a computational graph.
    '''
    def __init__(self,f):
        self.func = f

    def forward(self,A,result):
        return self.func(A,result)

    def backward(self,A,result):
        return self.func(A,result,backward=True)

class Loss_function():
    '''
    Class of a function that can be used as a loss function in a computational graph.
    '''
    def __init__(self,f):
        self.func = f

    def forward(self,X,Y,result):
        self.func(X,Y,result)

    def backward(self,X,Y,result):
        self.func(X,Y,result,backward=True)

#Definitions of most useful functions
relu = Function(cuda_backend.relu)
sigmoid = Function(cuda_backend.sigmoid)
mse = Loss_function(cuda_backend.mse)

#Dictionary mapping strings with function names to the functions themselves
function_dict = {'relu':relu,'sigmoid':sigmoid,'mse':mse}
