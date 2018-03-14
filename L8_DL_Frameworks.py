import numpy as np
np.random.seed(0)
import tensorflow as tf

N, D = 3000, 4000

with tf.device ('/gpu:0): # one sentence to change the code running on CPU or GPU
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    z = tf.placeholder(tf.float32)
				
    a = x + y
    b = a + z
    c = tf.reduce_sum(b)
                
								
