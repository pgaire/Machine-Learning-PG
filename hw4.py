import numpy as np 
import scipy.optimize as opt

# train data
train_data = np.loadtxt('./train.csv', delimiter =',',usecols = range(5))
#test data    
test = np.loadtxt('./test.csv', delimiter =',',usecols = range(5))

# get vector x and y for both train and test datasets
x_training = train_data[:,:-1]
train_1 = np.ones(x_training.shape[0])
train_nw = np.column_stack((x_training,train_1))
y_training = train_data[:,-1]
y_training = 2 * y_training - 1 

X_test = test[:,:-1]
one_test = np.ones(X_test.shape[0])
D_test = np.column_stack((X_test,one_test))
Y_test = test[:,-1]
Y_test = 2 * Y_test - 1

r = 0.1 
a = 0.1 
T = 100


c_value = np.array([(100/873), (500/873),(700/873)])
gamma_value = np.array([0.1, 0.5, 1, 5, 100])

def two_a(x, y, C, r=0.1):
	no_feature = x.shape[1]
	no_sample = x.shape[0]
	w = np.zeros(no_feature)
	idx = np.arange(no_sample)
	for t in range(T):
		np.random.shuffle(idx)
		x = x[idx,:]
		y = y[idx]
		for i in range(no_sample):
			temp = y[i] * np.dot(w, x[i])
			g = np.copy(w)
			g[no_feature-1] = 0
			if temp <= 1:
					g = g - C * no_sample * y[i] * x[i,:]
			r = r / (1 + r / a * t)
			w = w - r * g
	return w


def two_b(x, y, C, r=0.1):
	no_feature = x.shape[1]
	no_sample = x.shape[0]
	w = np.zeros(no_feature)
	idx = np.arange(no_sample)
	for t in range(T):
		np.random.shuffle(idx)
		x = x[idx,:]
		y = y[idx]
		for i in range(no_sample):
			temp = y[i] * np.dot(w, x[i])
			g = np.copy(w)
			g[no_feature-1] = 0
			if temp <= 1:
					g = g - C * no_sample * y[i] * x[i,:]
			r = r / (1 + t)
			w = w - r * g
	return w

def con(alpha,y):
	t = np.matmul(np.reshape(alpha,(1, -1)), np.reshape(y,(-1,1)))
	return t[0]


def obj(alpha, x, y):
	l = 0
	l = l - np.sum(alpha)
	ayx = np.multiply(np.multiply(np.reshape(alpha,(-1,1)), np.reshape(y, (-1,1))), x)
	l = l + 0.5 * np.sum(np.matmul(ayx, np.transpose(ayx)))
	return l


def fun_dual(x, y, C):
	no_sample = x.shape[0]
	bnds = [(0, C)] * no_sample
	cons = ({'type': 'eq', 'fun': lambda alpha: con(alpha, y)})
	alpha0 = np.zeros(no_sample)
	res = opt.minimize(lambda alpha: obj(alpha, x, y), alpha0,  method='SLSQP', bounds=bnds,constraints=cons, options={'disp': False})
	
	w = np.sum(np.multiply(np.multiply(np.reshape(res.x,(-1,1)), np.reshape(y, (-1,1))), x), axis=0)
	idx = np.where((res.x > 0) & (res.x < C))
	b =  np.mean(y[idx] - np.matmul(x[idx,:], np.reshape(w, (-1,1))))
	w = w.tolist()
	w.append(b)
	w = np.array(w)
	return w


def gaussian_kernl(x1, x2, gamma):
	m1 = np.tile(x1, (1, x2.shape[0]))
	m1 = np.reshape(m1, (-1,x1.shape[1]))
	m2 = np.tile(x2, (x1.shape[0], 1))
	k = np.exp(np.sum(np.square(m1 - m2),axis=1) / -gamma)
	k = np.reshape(k, (x1.shape[0], x2.shape[0]))
	return k

def obj_gk(alpha, k, y):
	l = 0
	l = l - np.sum(alpha)
	ay = np.multiply(np.reshape(alpha,(-1,1)), np.reshape(y, (-1,1)))
	ayay = np.matmul(ay, np.transpose(ay))
	l = l + 0.5 * np.sum(np.multiply(ayay, k))
	return l


def train_gaussian_krnl(x, y, C, gamma):
	no_sample = x.shape[0]
	bnds = [(0, C)] * no_sample
	cons = ({'type': 'eq', 'fun': lambda alpha: con(alpha, y)})
	alpha0 = np.zeros(no_sample)
	k = gaussian_kernl(x, x, gamma)
	res = opt.minimize(lambda alpha: obj_gk(alpha, k, y), alpha0,  method='SLSQP', bounds=bnds,constraints=cons, options={'disp': False})
	w = np.sum(np.multiply(np.multiply(np.reshape(res.x,(-1,1)), np.reshape(y, (-1,1))), x), axis=0)
	return res.x,w


def prdct_gaussian_krnl(alpha, x0, y0, x, gamma):
	k = gaussian_kernl(x0, x, gamma)
	k = np.multiply(np.reshape(y0, (-1,1)), k)
	y = np.sum(np.multiply(np.reshape(alpha, (-1,1)), k), axis=0)
	y = np.reshape(y, (-1,1))
	y[y > 0] = 1
	y[y <=0] = -1
	return y




for C in c_value:
	w = two_a(train_nw, y_training, C, r)
	w = np.reshape(w, (5,1))

	pred = np.matmul(train_nw, w)
	pred[pred > 0] = 1
	pred[pred <= 0] = -1
	train_err = np.sum(np.abs(pred - np.reshape(y_training,(-1,1)))) / 2 / y_training.shape[0]

	pred = np.matmul(D_test, w)
	pred[pred > 0] = 1
	pred[pred <= 0] = -1

	err_test = np.sum(np.abs(pred - np.reshape(Y_test,(-1,1)))) / 2 / Y_test.shape[0]
	print('training error is: ', train_err) 
	print('test error is: ', err_test)
	w = np.reshape(w, (1,-1))
	print("the weights function is:", w)

print()

for C in c_value:
	w1 = two_b(train_nw, y_training, C, r)
	w1 = np.reshape(w1, (5,1))

	pred = np.matmul(train_nw, w1)
	pred[pred > 0] = 1
	pred[pred <= 0] = -1
	train_err = np.sum(np.abs(pred - np.reshape(y_training,(-1,1)))) / 2 / y_training.shape[0]

	pred = np.matmul(D_test, w1)
	pred[pred > 0] = 1
	pred[pred <= 0] = -1

	err_test = np.sum(np.abs(pred - np.reshape(Y_test,(-1,1)))) / 2 / Y_test.shape[0]
	print('training error is: ', train_err)
	print('testing error is: ', err_test)
	w1 = np.reshape(w1, (1,-1))
	print("the weights function is:",w1)

print()

for C in c_value:
	w2 = fun_dual(train_nw[:,[x for x in range(4)]] ,y_training, C)

	w2 = np.reshape(w2, (5,1))

	pred = np.matmul(train_nw, w2)
	pred[pred > 0] = 1
	pred[pred <= 0] = -1
	train_err = np.sum(np.abs(pred - np.reshape(y_training,(-1,1)))) / 2 / y_training.shape[0]

	pred = np.matmul(D_test, w2)
	pred[pred > 0] = 1
	pred[pred <= 0] = -1

	err_test = np.sum(np.abs(pred - np.reshape(Y_test,(-1,1)))) / 2 / Y_test.shape[0]
	print('training error is: ', train_err)
	print('testing error is: ', err_test)
	print(w2)

print()


C=c_value[0]
for gamma in gamma_value:
	print('gamma: ', gamma, 'C:', C)
	alpha,w = train_gaussian_krnl(train_nw[:,[x for x in range(4)]] ,y_training, C, gamma)
		# train 
	y = prdct_gaussian_krnl(alpha, train_nw[:,[x for x in range(4)]], y_training, train_nw[:,[x for x in range(4)]], gamma)
	train_err = np.sum(np.abs(y - np.reshape(y_training,(-1,1)))) / 2 / y_training.shape[0]

		# test
	y = prdct_gaussian_krnl(alpha, train_nw[:,[x for x in range(4)]], y_training, X_test[:,[x for x in range(4)]], gamma)
	err_test = np.sum(np.abs(y - np.reshape(Y_test,(-1,1)))) / 2 / Y_test.shape[0]
	print(w)
	print('training error is: ', train_err)
	print('testing error is: ', err_test)


C=c_value[1]
for gamma in gamma_value:
	print('gamma: ', gamma, 'C:', C)
	alpha = train_gaussian_krnl(train_nw[:,[x for x in range(4)]] ,y_training, C, gamma)
		# train 
	y = prdct_gaussian_krnl(alpha, train_nw[:,[x for x in range(4)]], y_training, train_nw[:,[x for x in range(4)]], gamma)
	train_err = np.sum(np.abs(y - np.reshape(y_training,(-1,1)))) / 2 / y_training.shape[0]

		# test
	y = prdct_gaussian_krnl(alpha, train_nw[:,[x for x in range(4)]], y_training, X_test[:,[x for x in range(4)]], gamma)
	err_test = np.sum(np.abs(y - np.reshape(Y_test,(-1,1)))) / 2 / Y_test.shape[0]
	print('training error is: ', train_err)
	print('testing error is: ', err_test)

C=c_value[2]
for gamma in gamma_value:
	print('gamma: ', gamma, 'C:', C)
	alpha = train_gaussian_krnl(train_nw[:,[x for x in range(4)]] ,y_training, C, gamma)
		# train 
	y = prdct_gaussian_krnl(alpha, train_nw[:,[x for x in range(4)]], y_training, train_nw[:,[x for x in range(4)]], gamma)
	train_err = np.sum(np.abs(y - np.reshape(y_training,(-1,1)))) / 2 / y_training.shape[0]

		# test
	y = prdct_gaussian_krnl(alpha, train_nw[:,[x for x in range(4)]], y_training, X_test[:,[x for x in range(4)]], gamma)
	err_test = np.sum(np.abs(y - np.reshape(Y_test,(-1,1)))) / 2 / Y_test.shape[0]
	print('training error is: ', train_err)
	print('testing error is: ', err_test)