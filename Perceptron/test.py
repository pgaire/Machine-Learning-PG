
import numpy as np
import pandas as pd
data_train = pd.read_csv('train.csv', header=None)
data_test = pd.read_csv('test.csv', header=None)
columns = ['variance', 'skewness', 'curtosis', 'entropy', 'label']
features = columns[:-1]
output = columns[-1]
data_train.columns = columns
data_test.columns = columns
    
input_training = data_train.iloc[:, :-1].values
input_testing = data_test.iloc[:, :-1].values
train_labels = data_train.iloc[:, -1].values
test_labels = data_test.iloc[:, -1].values

class Perceptron_voted(object):
    def __init__(self, input_no, rate=0.05):
    
        self.rate = rate   # learning rate
        self.weights = np.zeros(input_no + 1)  # initialize weights to zero
        #self.weights_set = [np.zeros(input_no + 1)]
        self.a = np.zeros(input_no + 1)
        self.C=[0]

    

    def train(self, input_training, labels):
        # trains perceptron weights on training dataset
        weights = np.zeros(input_training.shape[1] + 1)
        weights_set = [np.zeros(input_training.shape[1]+1)]
        labels = np.expand_dims(labels, axis=1)
        data = np.hstack((input_training, labels))
        m = 0
        
        for e in range(10):
            np.random.shuffle(data)
            for row in data:
                inputs = row[:-1]
                label = row[-1]
                prediction = self.predict(inputs, weights)
                error = label - prediction
                if error:
                    weights[1:] += self.rate * (label - prediction) * inputs
                    weights[0] += self.rate * (label - prediction)
                    weights_set.append(np.copy(weights))

                    self.C.append(1)
                    m += 1

                else:
                    self.C[m] += 1

        self.weights = weights
        self.weights_set = weights_set

        return self.weights

    def evaluate(self, input_testing, labels):
        error = []
        n_weights = len(self.weights_set)
        for inputs, label in zip(input_testing, labels):
            predictions = []
            for k in range(n_weights):
                pred = self.predict(inputs, weights=self.weights_set[k])
                if not pred:
                    pred = -1
                predictions.append(self.C[k]*pred)

            prediction = np.sign(sum(predictions))
            if prediction == -1:
                prediction = 0

            error.append(np.abs(label-prediction))

        return sum(error) / float(input_testing.shape[0])
    def predict(self, inputs, weights):
        summation = np.dot(inputs, weights[1:]) + weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation



class Perceptron_standard(object):

    def __init__(self, input_no, rate=0.05):
       
        self.rate = rate   # rate of learning
        self.weights = np.zeros(input_no + 1) 

    

    def train(self, input_training, labels):

        labels = np.expand_dims(labels, axis=1)
        data = np.hstack((input_training, labels))
        for e in range(10):

            np.random.shuffle(data)
            for row in data:
                inputs = row[:-1]
                label = row[-1]
                prediction = self.predict(inputs)
                self.weights[1:] += self.rate * (label - prediction) * inputs
                self.weights[0] += self.rate * (label - prediction)

        return self.weights

    def evaluate(self, input_testing, labels):

        error = []
        for inputs, label in zip(input_testing, labels):
            prediction = self.predict(inputs)
            error.append(np.abs(label-prediction))

        return sum(error) / float(input_testing.shape[0])
    def predict(self, inputs):
        sum = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if sum > 0:
            label = 1
        else:
            label = 0
        return label


class Perceptron_Avg(object):

    def __init__(self, input_no, rate=0.05):
    
        self.rate = rate   # rate of learning
        self.weights = np.zeros(input_no + 1)  # initializing weight vector to zero
        self.a = np.zeros(input_no + 1)

    def predict(self, inputs, weights):
        summation = np.dot(inputs, weights[1:]) + weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

    def train(self, input_training, labels):
        weights = np.zeros(input_training.shape[1] + 1)
        weights_set = [np.zeros(input_training.shape[1]+1)]
        labels = np.expand_dims(labels, axis=1)
        data = np.hstack((input_training, labels))
        m = 0
        for e in range(10):
            np.random.shuffle(data)
            for row in data:
                inputs = row[:-1]
                label = row[-1]
                prediction = self.predict(inputs, weights)
                error = label - prediction
                weights[1:] += self.rate * (label - prediction) * inputs
                weights[0] += self.rate * (label - prediction)
                self.a += np.copy(weights)
        self.weights = weights

        return self.a

    def evaluate(self, input_testing, labels):
        error = []
        for inputs, label in zip(input_testing, labels):
            prediction = self.predict(inputs, weights=self.a)
            error.append(np.abs(label-prediction))

        return sum(error) / float(input_testing.shape[0])
    def predict(self, inputs, weights):
        summation = np.dot(inputs, weights[1:]) + weights[0]
        if summation > 0:
            activation = 1
        else:
            activation = 0
        return activation

 # Standard Perceptron Testing
perceptron_s = Perceptron_standard(4)
perceptron_s.train(input_training, train_labels)
error_s = perceptron_s.evaluate(input_testing, test_labels)
print("Standard Perceptron Test Error: " + str(error_s))
weights = []
error = []
perceptron = Perceptron_standard(4)
weights.append(perceptron.train(input_training, train_labels))
error.append(perceptron.evaluate(input_testing, test_labels))

print(np.mean(weights, axis=0))

 # Voted Perceptron Testing
perceptron_v = Perceptron_voted(4)
perceptron_v.train(input_training, train_labels)
error_v = perceptron_v.evaluate(input_testing, test_labels)
print("Voted Perceptron Test Error: " + str(error_v))
weights = []
error = []
perceptron = Perceptron_voted(4)
weights.append(perceptron.train(input_training, train_labels))
error.append(perceptron.evaluate(input_testing, test_labels))
print(np.mean(weights, axis=0))

 # Average Perceptron Testing
perceptron_a = Perceptron_Avg(4)
perceptron_a.train(input_training, train_labels)
error_a = perceptron_a.evaluate(input_testing, test_labels)
print("Average Perceptron Test Error: " + str(error_a))
weights = []
error = []

perceptron = Perceptron_Avg(4)
weights.append(perceptron.train(input_training, train_labels))
error.append(perceptron.evaluate(input_testing, test_labels))
print(np.mean(weights, axis=0))