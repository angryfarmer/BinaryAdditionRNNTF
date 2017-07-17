import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0,"C:\\Users\\ANGRY\\Documents\\Projects\\Tensorflow\\BinaryAdditionRNN")
import generate_binary_data as gbd

input_dim = 2
output_dim = 1
L_1_nodes = 10
number_of_bits = 8

## Setup weights
X_input, Y_input = gbd.generate_binary_set(20000,number_of_bits)
# X_test, Y_test = gbd.manual_test_set(number_of_bits,[[2,2],[3,3],[124,35],[24,89]])
X_test, Y_test = gbd.generate_binary_set(20000,number_of_bits)
# print(np.shape(Y_input))
# print(np.shape(X_input))
X = tf.placeholder(tf.float32,[None,input_dim,number_of_bits])
# X = tf.concat([X,tf.zeros([1,tf.shape(X)[1],input_dim])])
W_1_i = tf.Variable(tf.truncated_normal([input_dim,L_1_nodes],stddev = 0.5, mean = 0),name = "W_1_input")
W_1_c = tf.Variable(tf.truncated_normal([input_dim,L_1_nodes],stddev = 0.5, mean = 0),name = "W_1_candidate")
W_1_f = tf.Variable(tf.truncated_normal([input_dim,L_1_nodes],stddev = 0.5, mean = 0),name = "W_1_forget")
W_1_o = tf.Variable(tf.truncated_normal([input_dim,L_1_nodes],stddev = 0.5, mean = 0),name = "W_1_output")
W_h_i = tf.Variable(tf.truncated_normal([L_1_nodes,L_1_nodes],stddev = 0.5, mean = 0),name = "W_h_input")
W_h_c = tf.Variable(tf.truncated_normal([L_1_nodes,L_1_nodes],stddev = 0.5, mean = 0),name = "W_h_candidate")
W_h_f = tf.Variable(tf.truncated_normal([L_1_nodes,L_1_nodes],stddev = 0.5, mean = 0),name = "W_h_forget")
W_h_o = tf.Variable(tf.truncated_normal([L_1_nodes,L_1_nodes],stddev = 0.5, mean = 0),name = "W_h_output")
W_cell_state = tf.Variable(tf.truncated_normal([L_1_nodes,L_1_nodes],stddev = 0.5, mean = 0),name = "W_cell_state")
W_2 = tf.Variable(tf.truncated_normal([L_1_nodes,output_dim],stddev = 0.5, mean = 0),name = "W_2")
y_ = tf.placeholder(tf.float32,[None,output_dim,number_of_bits])

## Setup feedforward and error
L_H_prev = tf.zeros([tf.shape(X)[0],L_1_nodes])
cell_state_prev = tf.zeros([tf.shape(X)[0],L_1_nodes])
error = tf.reduce_sum(L_H_prev)
for n in range(number_of_bits):
	X_bit = tf.squeeze(tf.slice(X,[0,0,number_of_bits - n - 1],[-1,-1,1]),[2])

	input_gate_1 = tf.sigmoid(tf.add(tf.matmul(X_bit,W_1_i),tf.matmul(L_H_prev,W_h_i)))
	candidate_value_1 = tf.tanh(tf.add(tf.matmul(X_bit,W_1_c),tf.matmul(L_H_prev,W_h_c)))
	forget_gate_1 = tf.sigmoid(tf.add(tf.matmul(X_bit,W_1_f),tf.matmul(L_H_prev,W_h_f)))
	cell_state_1 = tf.add(tf.multiply(input_gate_1,candidate_value_1),tf.multiply(forget_gate_1,cell_state_prev))
	output_1 = tf.sigmoid(tf.add(tf.matmul(cell_state_1,W_cell_state),tf.add(tf.matmul(X_bit,W_1_o),tf.matmul(L_H_prev,W_h_o))))
	L_1 = tf.multiply(output_1,tf.tanh(cell_state_1))
	# L_H_prev = tf.concat([tf.reshape(L_1,[1,tf.shape(L_1)[0],tf.shape(L_1)[1]]),L_H_append],[0])
	cell_state_prev = tf.identity(cell_state_1)	
	L_H_prev = tf.identity(L_1)

	L_2 = tf.sigmoid(tf.matmul(L_1,W_2))
	y_bit = tf.squeeze(tf.slice(y_,[0,0,number_of_bits - n - 1],[-1,-1,1]),[1])
	if(n == 0):
		result = tf.round(L_2)
	else:
		result = tf.concat([tf.round(L_2),result],1)
	error = tf.add(error,tf.reduce_sum(tf.square(tf.subtract(y_bit,L_2))))

result_errors = tf.reduce_sum(tf.abs(tf.subtract(result,tf.squeeze(y_,[1]))),1)
result_errors = tf.where(tf.equal(result_errors,tf.zeros(tf.shape(result_errors))),tf.zeros(tf.shape(result_errors),dtype = tf.int32),tf.ones(tf.shape(result_errors),dtype = tf.int32))
# accuracy = tf.divide(tf.reduce_sum(result_errors),tf.shape(result_errors)[0])

## Training
Alpha = 0.01
decay_steps = 500
decay_rate = 0.096
global_step = tf.Variable(0,trainable = False,name = "global_step")
learning_rate = tf.train.exponential_decay(Alpha,global_step,decay_steps,decay_rate,staircase = True)

# print('pass')
train_step = tf.train.AdamOptimizer(learning_rate).minimize(error)

sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())

# print(sess.run(y_bit))
for n in range(1000):
	if(n % 100 == 0):
		print(sess.run(error, feed_dict = {X: X_input, y_: Y_input}))
	sess.run(train_step, feed_dict = {X: X_input, y_: Y_input})

saver = tf.train.Saver()
saver.save(sess,"./model/binary_addition_rnn_lstm_model")
print(sess.run(tf.reduce_sum(result_errors),feed_dict = {X: X_test, y_: Y_test}))
# print(sess.run(result,feed_dict = {X: X_test}))
# print(Y_test)

