# Making a Artificial Neural Network with 2X3X1 cells 
# Adding an input cell with '1' for representing the bias
# Using Identity Fn as the activation function for output

# Predicting Mom's Emotion with AI when I am playing a game or studying.
# Input (x1)    Input(x2)     Output (y)
# Playing Game  Studying      Mom's Emotion  
#      0           0           -1 (little angry) 
#      0           1            3 (very happy)
#      1           0           -3 (very angry)
#      1           1            1 (little happy)
#      0.5         0.5              ??
# If I make the time of studying and Playing in a half, Would Mom be angry or not? 

import numpy as np
from random import random

alpha = 0.3   # Learning Rate
epoch = 1000  # Iteration of Learning

# Initializing of weights 
wt = []    # Making a vacat list at first, Adding values using 'append' later 

for i in range(13):  # Need 13 weight values in 2X3X1 NN
    w = np.random.rand()  # Setting initial weights with random values (0~1)
    wt.append(w)

# Sigmoid Activation Function
def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

# Input and Teaching data (Datasets)
input_data = np.array([[0,0], [0,1], [1,0], [1,1]])
teaching_data = np.array([[-1], [3], [-3], [1]])

# Learning with Datasets
for n in range(1, epoch+1): # Iterating from 1 to 'epoch'
    for i in range(len(input_data)): 
        x1 = input_data[i][0]   # value of the 1st column in 'i'th row
        x2 = input_data[i][1]   # value of the 2nd column in 'i'th row
        t  = teaching_data[i]   # value of 'i'th row
        ########## Calculating Output (Forward) #########
        u1 = sigmoid(wt[0]*x1 + wt[3]*x2 + wt[6])
        u2 = sigmoid(wt[1]*x1 + wt[4]*x2 + wt[7])
        u3 = sigmoid(wt[2]*x1 + wt[5]*x2 + wt[8])
        y  = wt[9]*u1 + wt[10]*u2 + wt[11]*u3 + wt[12]
        ######## Backpropagation (Backward) ########
        E = 0.5 * (y - t)**2   # Loss Fn (Mean Squre Error)
        dE_dw_0 = (y-t)*wt[9]* (1-u1)*u1*x1
        dE_dw_1 = (y-t)*wt[10]*(1-u2)*u2*x1
        dE_dw_2 = (y-t)*wt[11]*(1-u3)*u3*x1
        dE_dw_3 = (y-t)*wt[9]* (1-u1)*u1*x2
        dE_dw_4 = (y-t)*wt[10]*(1-u2)*u2*x2
        dE_dw_5 = (y-t)*wt[11]*(1-u3)*u3*x2
        dE_dw_6 = (y-t)*wt[9]* (1-u1)*u1
        dE_dw_7 = (y-t)*wt[10]*(1-u2)*u2 
        dE_dw_8 = (y-t)*wt[11]*(1-u3)*u3 
        dE_dw_9 =  (y-t)*u1 
        dE_dw_10 = (y-t)*u2 
        dE_dw_11 = (y-t)*u3
        dE_dw_12 = (y-t)
        ########## Updating of weights (Gradient Descent) #########
        wt[0] = wt[0] - alpha * dE_dw_0
        wt[1] = wt[1] - alpha * dE_dw_1
        wt[2] = wt[2] - alpha * dE_dw_2
        wt[3] = wt[3] - alpha * dE_dw_3
        wt[4] = wt[4] - alpha * dE_dw_4
        wt[5] = wt[5] - alpha * dE_dw_5
        wt[6] = wt[6] - alpha * dE_dw_6
        wt[7] = wt[7] - alpha * dE_dw_7
        wt[8] = wt[8] - alpha * dE_dw_8
        wt[9] = wt[9] - alpha * dE_dw_9
        wt[10] = wt[10] - alpha * dE_dw_10
        wt[11] = wt[11] - alpha * dE_dw_11
        wt[12] = wt[12] - alpha * dE_dw_12

    print("{} EPOCH-ERROR: {}".format(n, E))

# Test: with new input value 'x', making prediction (=forward)
x1 = 0.5          # Time for playing a game
x2 = 0.4          # Time for studying
u1 = sigmoid(wt[0]*x1 + wt[3]*x2 + wt[6])
u2 = sigmoid(wt[1]*x1 + wt[4]*x2 + wt[7])
u3 = sigmoid(wt[2]*x1 + wt[5]*x2 + wt[8])
y  = wt[9]*u1 + wt[10]*u2 + wt[11]*u3 + wt[12]
print("My mom would be angry?")
print("Game:{} hour, Study:{} hour --> Mom's Emotion:{}".format(x1, x2, y))

if y > 0:
    print("Lucky! Mom would not be angry. Let's enjoy game together")
else:
    print("Oh My God! Mom would be angry. I should increase my study time.")
print("")