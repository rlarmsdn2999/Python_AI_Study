# General NN with iXjXk cells : Programming with the formula of 'EDI' and 'EDW DI'
# Output values should be between 0 and 1 because the output cells activated by sigmoid 

# Predict COVID19 or Cold? 
# 疑似新冠病毒的症状是如下；
#  Fever, Sense-,  Cough, Chest-pain    Cold/Flu  COVID19  
#         Disorder    
#   1        0       0        1            0         1
#   1        0       0        0            0.5      0.5
#   0        0       1        1            1         0
#   0        1       0        0            0         0
#   1        1       0        0            0         1
#   0        1       0        1            0        0.5
#   0        0       1        0            1         0

import numpy as np
from random import random
import matplotlib.pyplot as plt

######################### Hyper Parameter ######################
alpha = 0.1         # Learning Rate
epoch = 3000        # Iteration for Learning
n_hidden = 4        # Number of Node in Hidden Layer

###################  Weights Initialization #################### 
wt = []           # Vacant array for weights
bs = []           # Vacant array for bias
def init_weight(n_input, n_output):
    global wt, bs
    for i in range(n_input*n_hidden + n_hidden*n_output):
        w = np.random.rand()
        wt.append(w)
    for i in range(n_hidden + n_output):
        w = np.random.rand()
        bs.append(w)

################## Sigmoid activation function ################## 
def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

################# Calculating Output (Forward) ###################
def forward(x, n_output):
    u = []; y = []  # u: ouput in hidden layer, y: output in output layer
    n_input = len(x)

    # Output in hidden layer
    for j in range(n_hidden):        # Number of Node in hidden layer
        sum = 0
        for n in range(n_input):     # Number of Input
        # In case of 2X3X2: u1=s(w0x0+w3x1+b0), u2=s(w1x0+w4x1+b1), u3=s(w2x0+w5x1+b2)
            tmp = wt[n*n_hidden+j] * x[n]      # calculating 'weight X input' at first
            sum = sum + tmp
        u.append(sigmoid(sum + bs[j]))   # Caculating Weighted Sum and Activating with Sigmoid

    # Output in output layer
    for k in range(n_output):
        sum = 0
        for n in range(n_hidden):
        # In case of 2X3X2: y0=s(w6u0+w8u1+w10u2+b3), y1=s(w7u0+w9u1+w11u2+b4)
            tmp = wt[n_input*n_hidden + n*n_output+k] * u[n]
            sum = sum + tmp                    # calculating 'weight X input' at first
        y.append(sigmoid(sum+bs[n_hidden+k]))  # Caculating Weighted Sum and Activating with Sigmoid

    return u, y

##################### Back Propagation (Backward) ####################
def backpropagate(x, u, y, t):
    dE_dw = []          # Vacant list of weight-gradient
    dE_db = []          # Vacant list of bias-gradient
    n_input = len(x); n_output = len(t)
    for i in range(n_input):
        for j in range(n_hidden):
            sum = 0
            for n in range(n_output):
                tmp = (y[n]-t[n])*(1-y[n])*y[n]*wt[n_input*n_hidden+j+n_hidden*n] # 'EDI DI'
                # At first, caculating 'EDW' (weight-gradient in hidden layer)
                sum = sum + tmp
            dE_dw.append(sum*(1-u[j])*u[j]*x[i])
                # 'Formula of Weight-Gradient' in hidden layer: 'EDW' x 'DI'

    for j in range(n_hidden):
        sum = 0
        for k in range(n_output):
            dE_dw.append((y[k]-t[k])*(1-y[k])*y[k]*u[j])  # 'EDI'
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

########## Updating Weights & Bias (Gradient Decent, Optimization) #########
def update_weight(dE_dw, dE_db):
    global wt, bs
    for i in range(len(wt)):
        wt[i] = wt[i] - alpha * dE_dw[i]
    for i in range(len(bs)):
        bs[i] = bs[i] - alpha * dE_db[i]

######################## Calculating Loss Function ########################
def calc_error(y, t):
    err = 0
    for i in range(len(t)):
        tmp = 0.5*(y[i]-t[i])**2  # Loss Function: Mean Square Error
        err = err + tmp
    return err

def error_graph(error):  # Visualizing transient error with graph
    plt.ylim(0.0, 0.5)   # range of y axis(0 ~ 0.5)
    plt.plot(np.arange(0, error.shape[0]), error)
    plt.show()

################## Learning by Artificial Neural Network  ##################
def train(X, T):

    error = np.zeros(epoch)            # Initialization of Loss Function (Error)

    n_input = X.shape[1]               # Number of Input Node
    n_output = T.shape[1]              # Number of Output Node

    # Initialization of Weights
    init_weight(n_input, n_output)

    ############ Train with Datasets (input and teaching datum) #############
    for n in range(epoch):                  # Iterating as many as epoch
        for i in range(X.shape[0]):         # Number of Input data
            x = X[i, :]                     # x: All Input data
            t = T[i, :]                     # t: All Teaching data

            ######### Calculating output (Forward) #########
            u, y = forward(x, n_output)

            ######### Backpropagation (Backward)  ##########
            dE_dw, dE_db = backpropagate(x, u, y, t)

            ####### Weight Update (Gradient Descent) #######
            update_weight(dE_dw, dE_db)

            ############### Calculating Error ##############
            error[n] = calc_error(y, t)
        print("{} EPOCH-ERROR: {}".format(n, error[n]))

    error_graph(error)

################### Prediction with this Neural Network ###################
def predict(x, n_output):
    u, y = forward(x, n_output)
    return u, y

if __name__ == '__main__':
############# Datasets (Input & Teaching data) for learning #################
    X = np.array([[1,0,0,1], [1,0,0,0], [0,0,1,1], [0,1,0,0], [1,1,0,0], [0,1,0,1], [0,0,1,0]]) # Input
    T = np.array([[0, 1], [0.5, 0.5], [1, 0], [0, 0], [0, 1], [0, 0.5], [1, 0]])                # Answer

    train(X, T)

    ######### Test (Prepare dataset for test and make prediction with NN) #########
    x = np.array([1, 0, 1, 0])       # Test dataset (Fever and Cough)
    u, y = predict(x, T.shape[1])    # Prediction with Test dataset (input data for test)
    print("Cold or Corona?")
    print("Fever, Cough -> Cold? : {:.2f} %, Corona Virus ? : {:.2f} % ".format(y[0]*100, y[1]*100))
    print("")