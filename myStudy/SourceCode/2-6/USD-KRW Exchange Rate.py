# A Program for forecasting of the exchange-rate between KRW and USD by LSTM
# Download the datasets from https://kr.investing.com/currencies/usd-krw-historical-data
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas
from keras.models import Sequential
from keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler   # py -m pip install sklearn
from sklearn.metrics import mean_squared_error
# Making the datasets with the raw data(csv) using "pandas"
# Deleting unnecessary columns in the csv file
# Using "natural logarithm (ln)" for reducing of sudden changes
dataframe = pandas.read_csv('USD_KRW rate change.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32') 
# Fix random seed for reproducibility
np.random.seed(7)    # A Seed for making of randomized numbers. Same seed makes same random numbers.
# Normalize the dataset (Setting to the number between 0 and 1)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset) 
# Split the datasets into train and test sets
train_size = int(len(dataset) * 0.7)        # Set the early 70% of datasets into train sets 
test_size = len(dataset) - train_size       # Set the late 30% of datasets into test sets
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]
# print(len(train), len(test))  # check the length of the train sets and the test sets

# Convert an array of values into a dataset matrix
def create_dataset(dataset, maxlen):    # "maxlen" is the previous step number for forcasting the next time range
    dataX, dataY = [], []               # when maxlen=3, X is the exchange rate of "t-2", "t-1", "t", Y is the rate of "t+1"
    for i in range(len(dataset)-maxlen-1):
        a = dataset[i:(i+maxlen), 0]
        dataX.append(a)
        dataY.append(dataset[i + maxlen, 0])
    return np.array(dataX), np.array(dataY)
 
# Reshape into X=t and Y=t+maxlen
maxlen = 3   # what if increase the steps to 20?
trainX, trainY = create_dataset(train, maxlen)
testX, testY = create_dataset(test, maxlen)
 
print (trainX[:10,:])
print (trainY[:10])
 
# Reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print(trainX[:10,:])    # check the shape of datasets

# create and fit the LSTM network
# 1 input layer, hidden layer with 4 LSTM blocks, 1 output layer(Dense(1))
model = Sequential()
model.add(LSTM(4, input_shape=(1, maxlen)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam') # default activation function: Sigmoid
model.fit(trainX, trainY, epochs=30, batch_size=1, verbose=1) # verbose=1: Visualize the learning process

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)
# invert predictions
# converting the predicted data into origin form(array), caculating errors with "inverse_transform()"
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate RMSE(root mean squared error) 
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))
 
# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[maxlen:len(trainPredict)+maxlen, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(maxlen*2)+1:len(dataset)-1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset), color ="g", label = "row")
plt.plot(trainPredictPlot,color="b", label="trainpredict")
plt.plot(testPredictPlot,color="m", label="testpredict")
plt.title('Prediction with USD-KRW Exchange Rate') 
plt.legend()
plt.show()