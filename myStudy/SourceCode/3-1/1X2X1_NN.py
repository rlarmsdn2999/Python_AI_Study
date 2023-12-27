# Making the simplest Neural Network with 1X2X1 cells 
# Make Prediction of a Caculation with Sigmoid function
# Values of output should be between 0 and 1 because the output cell activated by sigmoid  

import numpy as np

alpha = 1.0   # Learning Rate
epoch = 5000  # Number of Learning-Iteration

# Setting of initial weights and bias 
w1 = 1.0    # 1st initial weight in hidden layer 
w2 = -1.0   # 2nd initial weight in hidden layer
w3 = 2.0    # 1st initial weight in output layer
w4 = -2.0   # 2nd initial weight in output layer 
b1 = -1.0   # 1st initial bias in hidden layer
b2 = 1.0    # 2nd initial bias in hidden layer
b3 = 2.0    # initial bias in output layer

# Sigmoid Activation Function 
def sigmoid(x):
    y = 1 / (1 + np.exp(-x))
    return y

# Datasets for Learning (Input and teaching datum)
input_data = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4, 5])
teaching_data = []   # Set a vacant list at first
for i in input_data: # Add datum to the list with the caculated values of sigmoid fn
    teaching_data.append(sigmoid(i))

# NN Learning with Datasets
for n in range(1, epoch+1): # Iterating from 1 to 'epoch'
    # transfer each input_data between [0] and [9] to the input 'x' 
    # transfer each teaching_data to the teaching data (Correct answer) 't'
    for i in range(len(input_data)):  # len(input_data) = 10 
        x = input_data[i]
        t = teaching_data[i]
        ########## Output (forward) #########
        u1 = sigmoid(w1 * x + b1)
        u2 = sigmoid(w2 * x + b2)
        y = sigmoid(w3 * u1 + w4 * u2 + b3)
        ########## Backpropagation (backward) ########
        E = 0.5 * (y - t)**2              # Loss Function (Mean Square Error)
        dE_dw_3 = (y-t) * (1-y) * y * u1  # Gradient of the 1st weight in output layer
        dE_dw_4 = (y-t) * (1-y) * y * u2  # Gradient of the 2nd weight in output layer
        dE_db_3 = (y-t) * (1-y) * y       # gradient of Bias in output layer
        dE_dw_1 = (y-t) * (1-y) * y * w3 * (1-u1) * u1 * x # Gradient of 1st weight in hidden layer
        dE_dw_2 = (y-t) * (1-y) * y * w4 * (1-u2) * u2 * x # Gradient of 2nd weight in hidden layer
        dE_db_1 = (y-t) * (1-y) * y * w3 * (1-u1) * u1     # Gradient of 1st bias in hidden layer
        dE_db_2 = (y-t) * (1-y) * y * w4 * (1-u2) * u2     # Gradient of 2nd bias in hidden layer
        ########## Updating of Weights and Bias (gradient descent) #########
        w1 = w1 - alpha * dE_dw_1
        w2 = w2 - alpha * dE_dw_2
        w3 = w3 - alpha * dE_dw_3
        w4 = w4 - alpha * dE_dw_4
        b1 = b1 - alpha * dE_db_1
        b2 = b2 - alpha * dE_db_2
        b3 = b3 - alpha * dE_db_3

    print("{} EPOCH-ERROR: {}".format(n, E))   # Printing the value of Loss Fn (Error) in each Epoch

# Test: With new input value 'x', predicting and checking the correct answer
x = 0.5                              # input for test
u1 = sigmoid(w1 * x + b1)            # output from hidden layer (forward)
u2 = sigmoid(w2 * x + b2)            # output from hidden layer (forward)
y = sigmoid(w3 * u1 + w4 * u2 + b3)  # final output (forward)
print("")
print("Guess by a NN: {}".format(y))
print("Calculated Value: {}".format(sigmoid(x)))
print("")