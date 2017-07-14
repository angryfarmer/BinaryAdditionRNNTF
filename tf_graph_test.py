import tensorflow as tf 

A = tf.ones([1,3])
B = tf.identity(A)
A = tf.scalar_mul(2,A)
D = tf.add(B,A)

sess = tf.InteractiveSession()
print(sess.run(D))