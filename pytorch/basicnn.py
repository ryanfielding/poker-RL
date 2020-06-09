import torch
import numpy as np 

def activation(x):
    #signoid activation func
    return 1/(1+torch.exp(-x))

# Basic 1 layer nn
#random data
torch.manual_seed(7) #set random seed for unpredictability
features = torch.randn((1,5)) #inputs to nn, random normal variables, size 1 row 5 cols
weights = torch.randn_like(features) #weights of same shape as features
bias = torch.randn((1,1))
#feedforward for result (output for single neuron)
y = activation(torch.sum(features * weights) + bias)
#or these
y = activation((features * weights).sum() + bias)
y = activation(torch.mm(features, weights.view(5,1)) + bias)
print(y)

#2 layer nn
torch.manual_seed(7)
features = torch.randn((1,3))
n_input = features.shape[1] #num of input units
n_hidden = 2
n_output = 1
#weights
W1 = torch.randn(n_input, n_hidden)
W2 = torch.randn(n_hidden, n_output)
#biases
B1 = torch.randn((1, n_hidden))
B2 = torch.randn((1, n_output))
#output
h = activation(torch.mm(features, W1) + B1)
output = activation(torch.mm(h, W2) + B2)
print(output)

#convert to and from numpy
a = np.random.rand(4,3)
b = torch.from_numpy(a)
print(a)
print(b)
b.numpy()
b.mul_(2)
print(a)