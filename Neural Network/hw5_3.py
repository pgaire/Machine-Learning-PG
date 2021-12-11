import numpy as np 
import warnings

# train data
train_data = np.loadtxt('train.csv', delimiter =',',usecols = range(5))
#test data    
test_data= np.loadtxt('test.csv', delimiter =',',usecols = range(5))

# get vector x and y for both train and test datasets
train_x = train_data[:,:-1]
train_one = np.ones(train_x.shape[0])
D_train = np.column_stack((train_x,train_one))
train_y = train_data[:,-1]
train_y = 2 * train_y - 1 

test_x = test_data[:,:-1]
test_one = np.ones(test_x.shape[0])
test_D = np.column_stack((test_x,test_one))
test_y = test_data[:,-1]
test_y = 2 * test_y - 1


lr = 0.01
d = 0.1
T = 100

v_list = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]

def train_max( x, y, v, lr):
 sample_num = x.shape[0]
 dimension = x.shape[1]
 weight = np.zeros([1, dimension])
 idx = np.arange(sample_num)
 for t in range(T):
  np.random.shuffle(idx)
  x = x[idx,:]
  y = y[idx]
  for i in range(sample_num):
   x_i = x[i,:].reshape([1, -1])
   tmp = y[i] * np.sum(np.multiply(weight, x_i))
   g = - sample_num * y[i] * x_i / (1 + np.exp(tmp)) + weight / v

   lr = lr / (1 + lr / d * t)
   weight = weight - lr * g
 return weight.reshape([-1,1])


def train_model(x, y, lr):
 num_sample = x.shape[0]
 dimension = x.shape[1]
 weight = np.zeros([1, dimension])
 idx = np.arange(num_sample)
 for t in range(T):
  np.random.shuffle(idx)
  x = x[idx,:]
  y = y[idx]
  for i in range(num_sample):
   tmp = y[i] * np.sum(np.multiply(weight, x[i,:]))
   g = - num_sample * y[i] * x[i,:] / (1 + np.exp(tmp))
   lr = lr / (1 + lr / d * t)
   weight = weight - lr * g
 return weight.reshape([-1,1])

print("For question 3(a)")
print("variance\tTrain Error\tTest Error")
print()
for v in v_list:

 weight= train_max(D_train, train_y, v, lr)


 pred = np.matmul(D_train, weight)
 pred[pred > 0] = 1
 pred[pred <= 0] = -1
 train_err = np.sum(np.abs(pred - np.reshape(train_y,(-1,1)))) / 2 / train_y.shape[0]*100

 pred = np.matmul(test_D, weight)
 pred[pred > 0] = 1
 pred[pred <= 0] = -1

 test_err = np.sum(np.abs(pred - np.reshape(test_y,(-1,1)))) / 2 / test_y.shape[0]*100
 print(f"{v}\t\t{train_err:.8f}\t{test_err:.8f}")

print()
print("For question 3(b)")
print("variance\tTrain Error\tTest Error")
print()
for v in v_list:

 weight= train_model(D_train, train_y, lr)


 pred = np.matmul(D_train, weight)
 pred[pred > 0] = 1
 pred[pred <= 0] = -1
 train_err = np.sum(np.abs(pred - np.reshape(train_y,(-1,1)))) / 2 / train_y.shape[0]*100

 pred = np.matmul(test_D, weight)
 pred[pred > 0] = 1
 pred[pred <= 0] = -1

 test_err = np.sum(np.abs(pred - np.reshape(test_y,(-1,1)))) / 2 / test_y.shape[0]*100
 print(f"{v}\t\t{train_err:.8f}\t{test_err:.8f}")