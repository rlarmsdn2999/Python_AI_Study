# Making the artificial Neural Network with 2X3X2 cells
# The Neural Network for Logic Operation of 'AND' and 'XOR'
# The value of output should be between 0 and 1 because the output cells are activated by sigmoid fn 

import numpy as np
from random import random

alpha = 1.0   # Learning Rate
epoch = 5000  # Iteration of Learning

# Initializing of weights and bias 
wt = []              # vacant array(list) for weights
bs = []              # vacant array(list) for bias
for i in range(12):  # 12 weights are necessary in 2X3X2 NN
    w = np.random.rand()
    
    wt.append(w)
for i in range(5):   # 5 bias are necessary in 2X3X2 NN
    w = np.random.rand()
    bs.append(w)

# Sigmoid activation function 
def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

# Datasets (Input and Teaching data)
input_data = np.array([[0,0], [0,1], [1,0], [1,1]])     # Input data
teaching_data = np.array([[0,0], [0,1], [0,1], [1,0]])  # [AND, XOR]

# Training (Learning) with datasets
for n in range(1, epoch+1): # Iterating from 1 to 'epoch'
    for i in range(len(input_data)): 
        x1 = input_data[i][0]   # Values of the 1st column in 'i'th row
        x2 = input_data[i][1]   # Values of the 2nd column in 'i'th row
        t1 = teaching_data[i][0]
        t2 = teaching_data[i][1]
        ########## Caculating Output Value (Forward) #########
        u1 = sigmoid(wt[0]*x1 + wt[1]*x2 + bs[0])  # 1st output in hidden layer
        u2 = sigmoid(wt[2]*x1 + wt[3]*x2 + bs[1])  # 2nd output in hidden layer
        u3 = sigmoid(wt[4]*x1 + wt[5]*x2 + bs[2])  # 3rd output in hidden layer
        y1 = sigmoid(wt[6]*u1 + wt[7]*u2 + wt[8]*u3 + bs[3])   # 1st output in output layer
        y2 = sigmoid(wt[9]*u1 + wt[10]*u2 + wt[11]*u3 + bs[4]) # 2nd output in output layer
        ########## Backpropagation (Backward) ########
        E = 0.5 * (y1 - t1)**2 + 0.5 * (y2 - t2)**2  # Loss Function
        # E: Error,  D: Derivative, W: Weight, I: Input
        dE_dw_0 = ((y1-t1)*(1-y1)*y1*wt[6] + (y2-t2)*(1-y2)*y2*wt[9])* (1-u1)*u1*x1
        #            (E        D      W   +    E        D        W)         D     I  
        dE_dw_1 = ((y1-t1)*(1-y1)*y1*wt[7] + (y2-t2)*(1-y2)*y2*wt[10])*(1-u2)*u2*x1
        dE_dw_2 = ((y1-t1)*(1-y1)*y1*wt[8] + (y2-t2)*(1-y2)*y2*wt[11])*(1-u3)*u3*x1
        dE_dw_3 = ((y1-t1)*(1-y1)*y1*wt[6] + (y2-t2)*(1-y2)*y2*wt[9])* (1-u1)*u1*x2
        dE_dw_4 = ((y1-t1)*(1-y1)*y1*wt[7] + (y2-t2)*(1-y2)*y2*wt[10])*(1-u2)*u2*x2
        dE_dw_5 = ((y1-t1)*(1-y1)*y1*wt[8] + (y2-t2)*(1-y2)*y2*wt[11])*(1-u3)*u3*x2
        dE_dw_6 =  (y1-t1)*(1-y1)*y1*u1
        #             E        D      I
        dE_dw_7 =  (y1-t1)*(1-y1)*y1*u2
        dE_dw_8 =  (y1-t1)*(1-y1)*y1*u3
        dE_dw_9 =  (y2-t2)*(1-y2)*y2*u1 
        dE_dw_10 = (y2-t2)*(1-y2)*y2*u2 
        dE_dw_11 = (y2-t2)*(1-y2)*y2*u3

        dE_db_0 = ((y1-t1)*(1-y1)*y1*wt[6] + (y2-t2)*(1-y2)*y2*wt[9])*  (1-u1)*u1*1
        #              E       D      W         E         D     W           D    I(1) 
        dE_db_1 = ((y1-t1)*(1-y1)*y1*wt[7] + (y2-t2)*(1-y2)*y2*wt[10])* (1-u2)*u2*1
        dE_db_2 = ((y1-t1)*(1-y1)*y1*wt[8] + (y2-t2)*(1-y2)*y2*wt[11])* (1-u3)*u3*1
        dE_db_3 = (y1-t1)*(1-y1)*y1*1
        #             E       D    I(1)
        dE_db_4 = (y2-t2)*(1-y2)*y2*1
        ########## Updating weights & bias (Gradient Descent) #########
        wt[0] =  wt[0] -  alpha * dE_dw_0
        wt[1] =  wt[1] -  alpha * dE_dw_1
        wt[2] =  wt[2] -  alpha * dE_dw_2
        wt[3] =  wt[3] -  alpha * dE_dw_3
        wt[4] =  wt[4] -  alpha * dE_dw_4
        wt[5] =  wt[5] -  alpha * dE_dw_5
        wt[6] =  wt[6] -  alpha * dE_dw_6
        wt[7] =  wt[7] -  alpha * dE_dw_7
        wt[8] =  wt[8] -  alpha * dE_dw_8
        wt[9] =  wt[9] -  alpha * dE_dw_9
        wt[10] = wt[10] - alpha * dE_dw_10
        wt[11] = wt[11] - alpha * dE_dw_11
        bs[0] =  bs[0] -  alpha * dE_db_0
        bs[1] =  bs[1] -  alpha * dE_db_1
        bs[2] =  bs[2] -  alpha * dE_db_2
        bs[3] =  bs[3] -  alpha * dE_db_3
        bs[4] =  bs[4] -  alpha * dE_db_4

    print("{} EPOCH-ERROR: {}".format(n, E))

# Test: With Input value 'x1', 'x2', Predicting Logic Operation (AND, XOR)
x1 = 1                   # Change the value of 'x1' and 'x2' for new test
x2 = 0
u1 = sigmoid(wt[0]*x1 + wt[1]*x2 + bs[0])
u2 = sigmoid(wt[2]*x1 + wt[3]*x2 + bs[1])
u3 = sigmoid(wt[4]*x1 + wt[5]*x2 + bs[2])
y1 = sigmoid(wt[6]*u1 + wt[7]*u2 + wt[8]*u3 + bs[3])
y2 = sigmoid(wt[9]*u1 + wt[10]*u2 + wt[11]*u3 + bs[4])
print("AND - XOR")
print("Input:[{}, {}] --> Output: [{:.1f}, {:.1f}]".format(x1, x2, y1, y2))
print("")