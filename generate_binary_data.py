import numpy as np
import random

def binary_list(value,number_of_bits):
	binary_of_value = bin(value)[2:].zfill(number_of_bits)
	return list(map(int,binary_of_value))	

def generate_binary_set(number_of_values, number_of_bits):
	max_value = 2 ** ( number_of_bits - 1 )
	X = np.zeros([number_of_values,2,number_of_bits],dtype = np.float64)
	Y = np.zeros([number_of_values,1,number_of_bits],dtype = np.float64)
	for n in range(number_of_values):
		X_1 = random.randint(0,max_value)
		X_2 = random.randint(0,max_value)
		Y_ = X_1 + X_2
		X_1 = binary_list(X_1,number_of_bits)
		X_2 = binary_list(X_2,number_of_bits)
		Y_ = binary_list(Y_,number_of_bits)
		for m in range(number_of_bits):
			X[n,0,m] = X_1[m]
			X[n,1,m] = X_2[m]
			Y[n,0,m] = Y_[m]
	return X,Y

def manual_test_set(number_of_bits,test_array):
	X = np.zeros([len(test_array),2,number_of_bits],dtype = np.float64)
	Y = np.zeros([len(test_array),1,number_of_bits],dtype = np.float64)
	for n,values in enumerate(test_array):
		X_1 = values[0]
		X_2 = values[1]
		Y_ = X_1 + X_2
		X_1 = binary_list(X_1,number_of_bits)
		X_2 = binary_list(X_2,number_of_bits)
		Y_ = binary_list(Y_,number_of_bits)
		for m in range(number_of_bits):
			X[n,0,m] = X_1[m]
			X[n,1,m] = X_2[m]
			Y[n,0,m] = Y_[m]
	return X,Y		

# a = np.array([[1,2,3],[1,2]])
# # print(np.shape(a))
# x,y = generate_binary_set(10,8)
# print(type(x))
# print(x)
# print(x.dtype)
# print(x)
# print(x[:,0,:])
# print(x[:,1,:])
# print(y)