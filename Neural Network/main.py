import pandas as pd
import numpy as np
import neuralnetwork as NN
import network2 as net2

data_input = pd.read_csv('train.csv', header=None)

raw_data = data_input.values
column_no = raw_data.shape[1]
row_no = raw_data.shape[0]
x_training = np.copy(raw_data)
x_training[:,column_no - 1] = 1
y_training = raw_data[:, column_no - 1]
y_training = 2 * y_training - 1

test_data = pd.read_csv('test.csv', header=None)
raw_data = test_data.values
column_no = raw_data.shape[1]
row_no = raw_data.shape[0]
x_testing = np.copy(raw_data)
x_testing[:,column_no - 1] = 1
y_testing = raw_data[:, column_no - 1]
y_testing = 2 * y_testing - 1

gammas = np.array([0.01, 0.1, 0.5, 1, 2, 5, 10, 100])
data_input = x_training.shape[1]
data_output = 1

list_width = [5, 10, 25, 50, 100]
print("for the case of random weights")
print("w    training error    testing error")
for width in list_width:
    s = [data_input, width, width, data_output]
    model= NN.NN(s)

    model.train(x_training.reshape([-1, data_input]), y_training.reshape([-1,1]))
    prediction = model.fit(x_training)

    prediction[prediction > 0] = 1
    prediction[prediction <= 0] = -1
    error_train = np.sum(np.abs(prediction - np.reshape(y_training,(-1,1)))) / 2 / y_training.shape[0]

    prediction = model.fit(x_testing)
    prediction[prediction > 0] = 1
    prediction[prediction <= 0] = -1

    error_test = np.sum(np.abs(prediction - np.reshape(y_testing,(-1,1)))) / 2 / y_testing.shape[0]
    print(str(width)+'          '+ str(error_train)+ '         ' +str(error_test))


print('for the case of zero weights')
print("w    training error    testing error")
for width in list_width:
    s = [data_input, width, width, data_output]
    model= net2.NN2(s)

    model.train(x_training.reshape([-1, data_input]), y_training.reshape([-1,1]))
    prediction = model.fit(x_training)

    prediction[prediction > 0] = 1
    prediction[prediction <= 0] = -1
    error_train = np.sum(np.abs(prediction - np.reshape(y_training,(-1,1)))) / 2 / y_training.shape[0]

    prediction = model.fit(x_testing)
    prediction[prediction > 0] = 1
    prediction[prediction <= 0] = -1

    error_test = np.sum(np.abs(prediction - np.reshape(y_testing,(-1,1)))) / 2 / y_testing.shape[0]
    print(str(width)+'          '+ str(error_train)+ '         ' +str(error_test))