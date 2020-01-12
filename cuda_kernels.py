'''
Contains gpu kernels written in cuda that are compiled to machine code.
'''

#Import libraries
from numba import cuda, float32
import numpy as np
import math

#Thread blocks are of size (TPB,TPB) and any two threads on the same block share memory
TPB = 16

#GPU CODE
#GPU kernels compiled to machine code

@cuda.jit
def matrix_addition_kernel(A,B,C):  
    #Locate the thread position
    x, y = cuda.grid(2)

    #Quit if thread is outside the matrix
    if A.shape[0]<=x or A.shape[1]<=y:
        return

    C[x,y] = A[x,y] + B[x,y]
    
@cuda.jit
def matrix_element_wise_mul_kernel(A,B,C):
    #Locate the thread position
    x, y = cuda.grid(2)

    #Quit if thread is outside the matrix
    if A.shape[0]<=x or A.shape[1]<=y:
        return

    C[x,y] = A[x,y] * B[x,y]

@cuda.jit
def matrix_constant_mul_kernel(c,A,B):
    #Locate the thread position
    x,y = cuda.grid(2)

    #Quit if thread is outside the matrix
    if A.shape[0]<=x or A.shape[1]<=y:
        return
    
    B[x,y] = A[x,y] * c

@cuda.jit
def matrix_mul_kernel(A,B,C,A_transponse,B_transponse):

    #Locate the thread position
    x, y = cuda.grid(2)

    #Calculate the maximum index of the touching dimension
    if A_transponse: dim_max = A.shape[0]
    else: dim_max = A.shape[1]

    #Temporary variables in which parts of matrices A and B will be loaded
    tempA = cuda.shared.array(shape=(TPB, TPB), dtype=float32)
    tempB = cuda.shared.array(shape=(TPB, TPB), dtype=float32)

    #Locate the thread position within the block
    tx = cuda.threadIdx.x
    ty = cuda.threadIdx.y

    stride = TPB

    iterations = math.ceil(dim_max/stride)

    #Stores the total sum computed by that particular thread
    total_sum = 0

    for i in range(iterations):

        #Calculate the current position of the 'sliding window'
        a_position = ty + i*stride
        b_position = tx + i*stride

        #Load parts of matrices but only if not exceeding the maximum length of a dimension
        if(a_position<dim_max):
            if not A_transponse: tempA[tx,ty] = A[x, a_position]
            else:  tempA[tx,ty] = A[a_position,x]

        else: 
            tempA[tx,ty] = 0

        if(b_position<dim_max):
            if not B_transponse: tempB[tx,ty] = B[b_position,y]    
            else:  tempB[tx,ty] = B[y,b_position]
                
        else: 
            tempB[tx,ty] = 0

        cuda.syncthreads()

        #Perform matrix multiplication on the loaded parts
        for j in range(stride):
            total_sum += tempA[tx, j] * tempB[j,ty]

        cuda.syncthreads()
    
    #Save the result only if the thread is indide the matrix
    if x < C.shape[0] and y < C.shape[1]: C[x,y] = total_sum


@cuda.jit
def sum_loss_kernel(s,A,n):

    s[n,0] = 0
    for x in range(A.shape[0]):
        for y in range(A.shape[1]):
            s[n,0] += A[x,y]

@cuda.jit
def matrix_copy_kernel(A,B):

    #Locate the thread position
    x,y = cuda.grid(2)

    #Quit if thread is outside the matrix
    if A.shape[0]<=x or A.shape[1]<=y:
        return

    B[x,y] = A[x,y]


@cuda.jit
def relu_kernel(A,B):
    #Locate the thread position
    x, y = cuda.grid(2)

    #Quit if thread is outside the matrix
    if A.shape[0]<=x or A.shape[1]<=y:
        return

    if A[x,y] <= 0:
        B[x,y] = 0
    else:
        B[x,y] = A[x,y]

@cuda.jit
def relu_backward_kernel(A,B):
    #Locate the thread position
    x, y = cuda.grid(2)

    #Quit if thread is outside the matrix
    if A.shape[0]<=x or A.shape[1]<=y:
        return

    if A[x,y] < 0: 
        B[x,y] = 0
    else: 
        B[x,y] = 1

@cuda.jit
def sigmoid_kernel(A,B):
    #Locate the thread position
    x, y = cuda.grid(2)

    #Quit if thread is outside the matrix
    if A.shape[0]<=x or A.shape[1]<=y:
        return

    B[x,y] = 1/(1+math.exp(-A[x,y]))

@cuda.jit
def sigmoid_backward_kernel(A,B):
    #Locate the thread position
    x, y = cuda.grid(2)

    #Quit if thread is outside the matrix
    if A.shape[0]<=x or A.shape[1]<=y:
        return

    B[x,y] = math.exp(-A[x,y])/(1+math.exp(-A[x,y]))**2

@cuda.jit
def mse_kernel(A,B,C):
    #Locate the thread position
    x, y = cuda.grid(2)

    #Quit if thread is outside the matrix
    if A.shape[0]<=x or A.shape[1]<=y:
        return

    C[x,y] = ((A[x,y]-B[x,y])**2)/A.shape[0] 

@cuda.jit
def mse_backward_kernel(A,B,C):
    #Locate the thread position
    x, y = cuda.grid(2)

    #Quit if thread is outside the matrix
    if A.shape[0]<=x or A.shape[1]<=y:
        return

    C[x,y] = (2*(A[x,y]-B[x,y]))/A.shape[0]

