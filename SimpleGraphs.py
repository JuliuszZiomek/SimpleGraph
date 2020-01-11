'''
SimpleGraphs framework.
Provides:
- Graph class which represents a computational graph
- Network class which represents a neural network
'''
#Import libraries
from numba import cuda
import numpy as np
import math
import sys

#Import auxiliary scripts
import cuda_backend
import functions

def init_weights(m,n):
    '''
    Initialize weight matrix w of size (m,n) with random variables
    using Xavier initialization and returns it
    '''
    w = np.random.normal(size=(m,n))
    w /= np.sqrt(m+n)
    
    return cuda.to_device(w)

def create_gpu_matrix(m,n):
    '''
    Creates gpu matrix of size (m,n) initialized to zeros
    and returns a pointer to it
    '''
    return cuda.to_device(np.zeros((m,n)))


class Graph():
    '''
    Computational graph class. Can perfrom forward and backward propagation.
    '''
    
    def __init__(self):
        #Stores the structure of the graph:
        self.__structure = []

        #Size of the previous matrix multiplication
        self.__prev_size = None 

        #Size of the recently forward propagated data set(for memory management)
        self.__data_size = None

        #Stores the output of the graph
        self.__output = None

        self.__size = None

    
    def get_structure(self):
        return self.__structure.copy()

    def add(self,operation,parameter):
        '''
        Add a node to the front of the computational graph.
        - If operation is mat_mul then parameter should be a tuple describing the shape of weight matrix
        - If operation is activation then parameter should be the activation function
        '''

        if operation=='mat_mul':
            self.__structure.append(
            {'operation':operation,
             'weight':init_weights(parameter[0],parameter[1]),
             'gradient':create_gpu_matrix(parameter[0],parameter[1]),
             'input':None,
             'delta':None})

        if operation=='activation':
            self.__structure.append(
            {'operation':operation,
             'function':parameter,
             'gradient':None,
             'input':None,
             'delta':None})

        self.__size = len(self.__structure)

    def forward(self,X):
        '''
        Perfroms forward propagation of X through the graph and returns the value of output.
        '''
        #Rearange gpu memory if size of dataset has changed
        if self.__data_size != X.shape[0]:
            self.__data_size = X.shape[0]
            self.__rearange_gpu_memory()
            

        #Feed X as input to the first layer
        self.__structure[0]['input'] = X

        for no in range(self.__size):
            step = self.__structure[no]

            #If layer is the last one its output is the output of the whole graph 
            #otherwise it is the input of the next step
            if no==self.__size-1: 
                output = self.__output
            else: 
                output = self.__structure[no+1]['input']
                
            #Forward propagate through the step in graph
            if step['operation'] =='mat_mul':
                cuda_backend.matrix_mul(step['input'],step['weight'],output)

            if step['operation'] =='activation':
                step['function'].forward(step['input'],output)


        return self.__output


    def __rearange_gpu_memory(self):
        '''
        Private function for dealing with memory management.
        '''

        curr_data_shape = None

        #Go through every step in the graph and reserve memory for it on the gpu
        for no in range(self.__size):
            step = self.__structure[no]

            if step['operation'] == 'mat_mul':
                step['input'] = create_gpu_matrix(self.__data_size,step['weight'].shape[0])
                step['delta'] = create_gpu_matrix(self.__data_size,step['weight'].shape[1])
                curr_data_shape = step['weight'].shape[1]
            
            if step['operation'] == 'activation':
                step['input'] = create_gpu_matrix(self.__data_size,curr_data_shape)
                step['delta'] = create_gpu_matrix(self.__data_size,curr_data_shape)
                step['gradient'] = create_gpu_matrix(self.__data_size,curr_data_shape)


        self.__output = create_gpu_matrix(self.__data_size,curr_data_shape)


    def backward(self,delta):
        '''
        Performs backward propagation through the graph.
        Taking delta as the gradient on the top of the graph 
        '''
        
        #Feed delta parameter as the delta of the last layer
        step = self.__structure[self.__size-1]['delta'] = delta

        for i in range(self.__size-1,0,-1):

            step = self.__structure[i]
            prev_step = self.__structure[i-1]    
           
           #Backprop through the step in graph
            if step['operation']=='mat_mul':
                cuda_backend.matrix_mul(step['input'],step["delta"],step["gradient"],A_transpond=True)
                cuda_backend.matrix_mul(step["delta"],step['weight'],prev_step["delta"],B_transpond=True)
            
            if step['operation']=='activation':
                step['function'].backward(step['input'],step['gradient'])
                cuda_backend.matrix_element_wise_mul(step['delta'],step['gradient'],prev_step["delta"])    
        

        #Backprop through the fist step in the graph
        step = self.__structure[0]
        if step['operation']=='mat_mul': cuda_backend.matrix_mul(step['input'],step["delta"],step["gradient"],A_transpond=True)

        
    def update_weights(self,alpha):
        '''
        Updates weights in the graph, based on the gradients computed in last backpropagation.
        Before update all, gradients are multiplied by alpha
        '''

        for step in self.__structure:
            if step['operation'] =='mat_mul':
                cuda_backend.matrix_constant_mul(-alpha,step['gradient'],step['gradient'])
                cuda_backend.matrix_addition(step['weight'], step['gradient'],step['weight'])
                
class Network(Graph):
    '''
    Neural Network class.
    '''

    def __init__(self):
        super().__init__()

        #Stores the structure of the network
        self.__net = []

        #Size of the output of previous layer
        self.__prev_out = None

        self.cost_history = None


    def set_input_size(self,in_size):
        '''
        Sets the number of input features.
        '''
        self.__prev_out = in_size

    def add_layer(self,size,activation=None):
        '''
        Adds a new layer to the network.
        Argument size specifies how many neurons should the layer have
        and activation specifies the activation function.
        '''
        
        layer = {'name':'Dense%d'%(len(self.__net)),
                'size':size,
                'activation':activation}

        self.__net.append(layer)

        self.add('mat_mul',(self.__prev_out,size))

        if activation!=None:
            if activation not in functions.function_dict.keys():
                raise Exception("Function %s is not defined"%(activation))
            else:
                self.add('activation',functions.function_dict[activation])
        
        self.__prev_out = size

    def set_loss_function(self,lossfn):
        if lossfn not in functions.function_dict.keys():
            raise Exception("Function %s is not defined"%(lossfn))
        else:
            self.lossfn = functions.function_dict[lossfn]
    
    def get_cost_history(self):
        return self.cost_history.copy_to_host()

    def train(self,X,Y,alpha,maxit):
        '''
        Performs a supervised training of the network.
        - X input data
        - Y output data
        - alpha learning rate
        - maxit maximut number of iterations
        '''

        #Temporary variables
        delta = cuda.to_device(np.zeros_like(Y))
        loss = cuda.to_device(np.zeros_like(Y))
        cost = cuda.to_device(np.zeros(shape=(maxit,1)))

        sys.stdout.write('\n Starting supervised training of the network for %d iterations. \n \n'%(maxit))

        for it in range(maxit):

            h = self.forward(X)

            self.lossfn.forward(h,Y,loss)

            cuda_backend.sum_loss(cost,loss,it)
                                                                
            self.lossfn.backward(h,Y,delta)

            self.backward(delta)

            self.update_weights(alpha)

            sys.stdout.flush()
            sys.stdout.write(' Completed %d out of %d iteration. \r'%(it+1,maxit))

        self.cost_history = cost


