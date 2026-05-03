import numpy as np
import tensorflow as tf

print("NumPy version:", np.__version__)
print("TensorFlow version:", tf.__version__)

# simple tensor test
a = tf.constant([1, 2, 3])
print("Tensor test:", a)


# result: 
#numpy version loaded: 1.24.3 - compatible for TF2.13.0
#compatible for windows - tensorflow loaded: 2.13.0
