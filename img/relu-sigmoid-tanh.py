#!/usr/bin/env python3

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scienceplots

mpl.rcParams['text.usetex'] = True
plt.style.use('science')

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

def tanh(x):
    return np.tanh(x)

x = np.linspace(-5, 5, 100)

y_sigmoid = sigmoid(x)
y_relu = relu(x)
y_tanh = tanh(x)

plt.plot(x, y_sigmoid, label='Sigmoid', color='blue', alpha=0.6, linewidth=1)
plt.plot(x, y_relu, label='ReLU', color='red', alpha=0.6, linewidth=1)
plt.plot(x, y_tanh, label='tanh', color='green', alpha=0.6, linewidth=1)

# Set up the plot
plt.xlabel('x')
plt.ylabel('f(x)')
#plt.title('Sigmoid, ReLU, and tanh Functions')
plt.legend()

plt.savefig('sigmoid-relu-tanh.pdf', bbox_inches='tight')
#plt.show()
