# 1 Numpy
import numpy as np
np.random.seed(0)

N, D = 3, 4

x = np.random.rand(N, D)
y = np.random.rand(N, D)
z = np.random.rand(N, D)

a = x + y
b = a + z
c = np.sum(b)
	
grad_c = 1.0 
grad_b = grad_c * np.ones((N, D))
grad_a = grad_b.copy()
grad_z = grad_a.copy()
grad_x = grad_a * y
grad_y = grad_a * x

# 2 TensorFlow
import numpy as np
np.random.seed(0)
import tensorflow as tf

N, D = 3, 4

with tf.device ('/gpu:0): # one sentence to change the code running on CPU or GPU
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    z = tf.placeholder(tf.float32)
				
    a = x + y
    b = a + z
    c = tf.reduce_sum(b)

# one sentence to calculate gradients     
grad_x, grad_y, grad_z = tf.gradients(c, (x, y, z))

with tf.Session() as sess:
    values = {
        x: np.random.randn(N, D),
	y: np.random.randn(N, D),
	z: np.random.randn(N, D),
    }
    out = sess.run({c, grad_x, grad_y, grad_z},
		    feed_dict=values)
    c_val, grad_x_val, grad_y_val, grad_z_val = out

                
# 3 PyTorch
import torch
from torch.autograd import Variable
                
N, D = 3, 4
                
x = Variable(torch.randn(N, D).cuda(), requires_grad=True)
y = Variable(torch.randn(N, D).cuda(), requires_grad=True)
z = Variable(torch.randn(N, D).cuda(), requires_grad=True)
                
a = x + y
b = a + z
c = torch.sum(b)
	  
c.backward()
                
print(x.grad.data)
print(y.grad.data)
print(z.grad.data)
