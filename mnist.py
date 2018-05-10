from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Imports
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

# Our application logic will be added here

if __name__ == "__main__":
    tf.app.run()


'''
1. Convolutional layer #1: 32 5x5 filters, with ReLU activation
2. Pooling Layer #1: Max pooling with a 2x2 filter and stride of 2

3. Convolutional Layer #2: 64 5x5 filters, with ReLU activation
4. Polling Layer #2: max pooling with 2x2 filter and stride of 2

5. Dense layer #1: 1,024 neurons, with dropout regularization rate of 0.4
6. Dense Layer #2: 10 neurons, one for each digit target class (0-9).
'''
