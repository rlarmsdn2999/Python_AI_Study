# General NN with iXjXk cells : Programming with the formula of 'EDI' and 'EDW DI'
# Activated by Sigmoid. Including Normalization and Denormalization. Be able to use any number.
# Make a rule of operating 3 numbers and Predict the result. Which one is stronger among AI and human? 
# The rule in this program: (a,b,c) --> (a+c-b-2), (b+c-a-3)  
# Make 3 teams and play a game, A:Making a rule, B:Finding the rule, C:AI Prediction 

import numpy as np
from random import random
import matplotlib.pyplot as plt

######################### Hyper Parameter ######################
alpha = 0.2          # Learning Rate
epoch = 50000        # Iteration for Learning
n_hidden = 5         # Number of Node in Hidden Layer
min_error = 0.001   # Learning will be stopped if the error get to be smaller than this value

############ Weights Initialization ############# 
wt = []           # Vacant array for weights
bs = []           # Vacant array for bias
def init_weight(n_input, n_output):
    global wt, bs    # declare global variable
    for i in range(n_input*n_hidden + n_hidden*n_output):
        w = np.random.rand()
        wt.append(w)
    for i in range(n_hidden + n_output):
        w = np.random.rand()
        bs.append(w)

########### Sigmoid activation function ########## 
def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

################ Normalization and Denormalization ################
# Function to normalize a list (list of general numbers -> 0~1)
# List for learning: Set the list to lst1 and lst2 
# List for test: Set the list for learning to lst1, Set the list for test to lst2
def norm_list(lst1, lst2):  # lst1:List for benchmark, lst2:List for normalization
    normalized = []
    for value in lst2:
        normal_num = (value - min(lst1))/(max(lst1) - min(lst1)) # Eq for nomalization
        normalized.append(normal_num)
    return normalized

# Function to  denormalize a list (0~1 -> list of general numbers)
def denorm_list(lst1, lst2):  # lst1: list for normalization, lst2: list for denormalization
    denormalized = []
    for value in lst2:
        denormal_num = value * (max(lst1) - min(lst1)) + min(lst1) # Eq for denormalization
        denormalized.append(denormal_num)   
    return denormalized

################### Calculating Output (Forward) ##################
def forward(x, n_output):
    u = []; y = []   # u: ouput in hidden layer, y: output in output layer
    n_input = len(x)

    # Output in hidden layer
    for j in range(n_hidden):        # Number of Node in hidden layer
        sum = 0
        for n in range(n_input):     # Number of Input
            tmp = wt[n*n_hidden+j] * x[n]  # calculating 'weight X input' at first
            sum = sum + tmp
        u.append(sigmoid(sum + bs[j]))  # Caculating Weighted Sum and Activating with Sigmoid

    # Output in output layer
    for k in range(n_output):
        sum = 0
        for n in range(n_hidden):
            tmp = wt[n_input*n_hidden + n*n_output+k] * u[n]
            sum = sum + tmp
        y.append(sigmoid(sum+bs[n_hidden+k]))

    return u, y

#################### Back Propagation (Backward) ###################
def backpropagate(x, u, y, t):
    dE_dw = []          # Vacant list of weight-gradient
    dE_db = []          # Vacant list of bias-gradient
    n_input = len(x); n_output = len(t)

    for i in range(n_input):
        for j in range(n_hidden):
            sum = 0
            for n in range(n_output):
                tmp = (y[n]-t[n])*(1-y[n])*y[n]*wt[n_input*n_hidden+j+n_hidden*n] 
                # At first, caculating 'EDW' (weight-gradient in hidden layer)
                sum = sum + tmp
            dE_dw.append(sum*(1-u[j])*u[j]*x[i])
                # 'Formula of Weight-Gradient' in hidden layer: 'EDW' x 'DI'

    for j in range(n_hidden):
        sum = 0
        for k in range(n_output):
            dE_dw.append((y[k]-t[k])*(1-y[k])*y[k]*u[j])  # EDI
            # 'Formula of Weight-Gradient' in output layer: 'EDI'
            tmp = (y[k]-t[k])*(1-y[k])*y[k]*wt[n_input*n_hidden+j+n_hidden*k]
            # At first, caculating 'EDW' (bias-gradient in hidden layer)
            sum = sum + tmp
        dE_db.append(sum*(1-u[j])*u[j])
            # 'Formula of Bias-Gradient' in hidden layer: 'EDW' x 'DI' (Input=1)
    
    for i in range(n_output):
        tmp = (y[i]-t[i])*(1-y[i])*y[i]
        dE_db.append(tmp)
            # 'Formula of Bias-Gradient' in output layer: 'EDI' (Input=1)
    
    return dE_dw, dE_db

############## Updating Weights & Bias (Gradient Decent, Optimization) #############
def update_weight(dE_dw, dE_db):
    global wt, bs
    for i in range(len(wt)):
        wt[i] = wt[i] - alpha * dE_dw[i]
    for i in range(len(bs)):
        bs[i] = bs[i] - alpha * dE_db[i]

###################### Calculating Loss Function #######################
def calc_error(y, t):
    err = 0
    for i in range(len(t)):
        tmp = 0.5*(y[i]-t[i])**2   # Loss Function: Mean Square Error
        err = err + tmp
    return err

def error_graph(error):    # Visualizing transient error with graph
    plt.ylim(0.0, 0.02)    # range of y axis(0 ~ 0.02)
    plt.plot(np.arange(0, error.shape[0]), error)
    plt.show()

##################  Learning by Artificial Neural Network  ###############
def train(X, T):

    error = np.zeros(epoch)            # Initialization of Loss Function (Error)

    n_input = X.shape[1]               # Number of Input Node
    n_output = T.shape[1]              # Number of Output Node

    # Initialization of Weights
    init_weight(n_input, n_output)

    ########### Train with Datasets (input and teaching datum) ############
    for n in range(epoch):                  # Iterating as many as epoch
        for i in range(X.shape[0]):         # Number of Input data
            x = X[i, :]                     # x: All Input data
            t = T[i, :]                     # t: All Teaching data

            ### Calculating output (Forward) ########
            u, y = forward(x, n_output)

            ##### Backpropagation (Backward) #######
            dE_dw, dE_db = backpropagate(x, u, y, t)

            ### Weight Update (Gradient Descent) #####
            update_weight(dE_dw, dE_db)

            ######## Calculating Error ########
            error[n] = calc_error(y, t)
        print("{} EPOCH-ERROR: {}".format(n, error[n]))

        if error[n] < min_error: # Stop learning if the error get to be smaller than 'min_error'
            break

    error_graph(error)

################### Prediction with this Neural Network ###################
def predict(x, n_output):
    u, y = forward(x, n_output)  # Caculating output (Forward)
    return u, y                  # u: output in hidden layer, y: output in output layer

if __name__ == '__main__':
################# Datasets (Input & Teaching data) for learning ############
    X = np.array([[1,2,3], [1,2,7], [2,3,7],   # Input Data: 
                  [3,4,6], [3,5,7], [3,7,8],   # 3 Arbitrary Number from 0 to 9
                  [5,6,7], [5,7,9], [5,8,9]])      
    T = np.array([[-2,1],  [-6,5],  [-4,5],    # Teaching Data:
                  [-1,4],  [-1,6],  [0,9],     # 1st No = (1st)+(2nd)-(3rd)-2
                  [2,5],   [1,8],   [2,9]])     # 2nd No = (2nd)+(3rd)-(1st)-3
                           


    X1 = X.reshape(-1)  # Making X to 1-D Array for Normalization
    T1 = T.reshape(-1)  # Making T to 1-D Array for Normalization
    n_X = norm_list(X1, X1)     # Normalization of Input data
    n_T = norm_list(T1, T1)     # Normalization of Teaching data
    
    n_X = np.array(n_X)       # Converting to np.array to use 'reshape' fn
    n_T = np.array(n_T)       # Converting to np.array to use 'reshape' fn
    n_X = n_X.reshape(-1, X.shape[1])   # Back to the origin-shaped-array after normalization
    n_T = n_T.reshape(-1, T.shape[1])   # Back to the origin-shaped-array after normalization

    train(n_X, n_T)           # Learning with datasets (input & teaching data)

    ############ Test (Prepare dataset for test and make prediction with NN) ###############
    test = np.array([4,5,9])              # Test dataset
    n_test = norm_list(X1, test)          # Normalization of Test dataset
    u, y = predict(n_test, T.shape[1])    # Prediction with Test dataset

    denorm_y = denorm_list(T1, y)         # Denormalization of prediction-result
 
    print("Input Numbers: {}, {}, {}".format(test[0], test[1], test[2]))
    print("Prediction: {:.2f}, {:.2f}".format(denorm_y[0], denorm_y[1]))
    print("Correct Answer: {:.2f}, {:.2f}".format(test[0]+test[1]-test[2]-2, test[1]+test[2]-test[0]-3))
    print("")