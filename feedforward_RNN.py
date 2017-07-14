import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0,"C:\\Users\\ANGRY\\Documents\\Projects\\Tensorflow\\BinaryAdditionRNN")
import generate_binary_data as gbd

input_dim = 2
output_dim = 1
L_1_nodes = 20
number_of_bits = 15

X_test, Y_test = gbd.generate_binary_set(20000,number_of_bits)

X = tf.placeholder(tf.float32,[None,input_dim,number_of_bits])
W_1 = tf.Variable(tf.truncated_normal([input_dim,L_1_nodes],stddev = 0.5, mean = 0),name = "W_1")
W_h = tf.Variable(tf.truncated_normal([L_1_nodes,L_1_nodes],stddev = 0.5, mean = 0),name = "W_h")
W_2 = tf.Variable(tf.truncated_normal([L_1_nodes,output_dim],stddev = 0.5, mean = 0),name = "W_2")
y_ = tf.placeholder(tf.float32,[None,output_dim,number_of_bits])

## Setup feedforward and error
L_H_prev = tf.zeros([tf.shape(X)[0],L_1_nodes])
error = tf.reduce_sum(L_H_prev)
for n in range(number_of_bits):
	X_bit = tf.squeeze(tf.slice(X,[0,0,number_of_bits - n - 1],[-1,-1,1]),[2])
	L_1 = tf.sigmoid(tf.add(tf.matmul(X_bit,W_1),tf.matmul(L_H_prev,W_h)))
	L_H_prev = tf.identity(L_1)
	L_2 = tf.sigmoid(tf.matmul(L_1,W_2))
	y_bit = tf.squeeze(tf.slice(y_,[0,0,number_of_bits - n - 1],[-1,-1,1]),[1])
	if(n == 0):
		result = tf.round(L_2)
	else:
		result = tf.concat([tf.round(L_2),result],1)

result_errors = tf.reduce_sum(tf.abs(tf.subtract(result,tf.squeeze(y_,[1]))),1)
result_errors = tf.where(tf.equal(result_errors,tf.zeros(tf.shape(result_errors))),tf.zeros(tf.shape(result_errors),dtype = tf.int32),tf.ones(tf.shape(result_errors),dtype = tf.int32))

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess,"./model/binary_addition_rnn_model")

print(sess.run(tf.reduce_sum(result_errors),feed_dict = {X: X_test, y_: Y_test}))
print(sess.run(result,feed_dict = {X: X_test}))