'''
Script providing an API that allows to use gpu kernels in cuda_kernels.py script.
'''

#Import libraries
from numba import cuda, float32
import numpy as np
import math

#Import auxiliary files
import cuda_kernels

#Thread blocks are of size (TPB,TPB) and any two threads on the same block share memory
TPB = cuda_kernels.TPB

#CPU CODE
#API for high-level script to use

#Auxiliary functions:
def check_shapes(A,B):
    if A.shape != B.shape:
        raise Exception("Shapes not matching: (%d,%d) (%d,%d)"%
        (A.shape[0],A.shape[1],B.shape[0],B.shape[1]))
    
def divide_memory(x_size,y_size):
    threadsperblock = (TPB, TPB)
    blockspergrid_x = math.ceil(x_size / threadsperblock[1])
    blockspergrid_y = math.ceil(y_size / threadsperblock[0])
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    return threadsperblock, blockspergrid

#Basic matrix operations:
def matrix_addition(A,B,C):
    '''
    Adds matrices A and B and saves the result in C.
    A, B and C must have the same shape.
    '''

    check_shapes(A,B)
    check_shapes(A,C)

    threadsperblock, blockspergrid = divide_memory(A.shape[0],A.shape[1])

    cuda_kernels.matrix_addition_kernel[blockspergrid, threadsperblock](A,B,C)

def matrix_constant_mul(c,A,B):
    '''
    Multiplies matrix A by constant c and saves the result in B.
    A and B must have the same shape.
    '''

    threadsperblock, blockspergrid = divide_memory(A.shape[0],A.shape[1])

    cuda_kernels.matrix_constant_mul_kernel[blockspergrid, threadsperblock](c,A,B)

def matrix_element_wise_mul(A,B,C):
    '''
    Multiplies matrices A and B element-wise and saves the result in C.
    A, B and C must have the same shape.
    '''

    check_shapes(A,B)
    check_shapes(A,C)

    threadsperblock, blockspergrid = divide_memory(A.shape[0],A.shape[1])

    cuda_kernels.matrix_element_wise_mul_kernel[blockspergrid, threadsperblock](A,B,C)
    
def matrix_mul(A,B,C,A_transponse=False,B_transponse= False):
    '''
    Performs matrix multiplication on A and B and saves the result in C.
    A, B and C must have the same shape.
    '''
    
    if not A_transponse:
        A_touching_shape = A.shape[1]
    else:
        A_touching_shape = A.shape[0]

    if not B_transponse:
        B_touching_shape = B.shape[0]
    else:
        B_touching_shape = B.shape[1]

    if A_touching_shape != B_touching_shape:
        raise Exception("Shapes not matching: (%d,%d) (%d,%d)"%
        (A.shape[0],A.shape[1],B.shape[0],B.shape[1]))

    threadsperblock, blockspergrid = divide_memory(A.shape[0],B.shape[1])

    cuda_kernels.matrix_mul_kernel[blockspergrid, threadsperblock](A,B,C,A_transponse,B_transponse)

def sum_loss(cost,loss,n):
    '''
    Sum los and saves it into the nth column of cost.
    '''
    cuda_kernels.sum_loss_kernel[1,1](cost,loss,n)

def matrix_copy(A,B):
    '''
    Copies the contents of matrix A into matrix B.
    A and B must have the same shape.
    '''

    check_shapes(A,B)

    threadsperblock, blockspergrid = divide_memory(A.shape[0],A.shape[1])

    cuda_kernels.matrix_element_wise_mul_kernel[blockspergrid, threadsperblock](A,B)

#Functions
def relu(A,B,backward=False):
    '''
    Performs relu operation on A and saves the result in B.
    If backward=True calculates the derivative of relu on A.
    A, B and C must have the same shape.
    '''
    
    threadsperblock, blockspergrid = divide_memory(A.shape[0],A.shape[1])

    if backward==False:
        cuda_kernels.relu_kernel[blockspergrid, threadsperblock](A,B)
    else:
        cuda_kernels.relu_backward_kernel[blockspergrid, threadsperblock](A,B)


def sigmoid(A,B,backward=False):
    '''
    Performs sigmoid operation on A and saves the result in B.
    If backward=True calculates the derivative of sigmoid on A.
    A, B and C must have the same shape.
    '''

    threadsperblock, blockspergrid = divide_memory(A.shape[0],A.shape[1])

    if backward==False:
        cuda_kernels.sigmoid_kernel[blockspergrid, threadsperblock](A,B)
    else:
        cuda_kernels.sigmoid_backward_kernel[blockspergrid, threadsperblock](A,B)


def mse(A,B,C,backward=False):
    '''
    Calculates mean squared error between A and B and saves the result in C.
    If backward=True calculates the derivative of mean squared error on A and B.
    A, B and C must have the same shape.
    '''

    check_shapes(A,B)

    threadsperblock, blockspergrid = divide_memory(A.shape[0],A.shape[1])

    if backward==False:
        cuda_kernels.mse_kernel[blockspergrid, threadsperblock](A,B,C)
    else:
        cuda_kernels.mse_backward_kernel[blockspergrid, threadsperblock](A,B,C)

