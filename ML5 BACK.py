import numpy as np

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative of Sigmoid Function
def derivatives_sigmoid(x):
    return x * (1 - x)

# Input and output data
X = np.array(([2, 9], [1, 5], [3, 6]), dtype=float)  # Two inputs [sleep, study]
y = np.array(([92], [86], [89]), dtype=float)  # One output [Expected % in Exams]
X = X / np.amax(X, axis=0)  # Normalizing input data
y = y / 100  # Normalizing output data

# Variable initialization
epoch = 5000  # Setting training iterations
lr = 0.1  # Setting learning rate
inputlayer_neurons = 2  # Number of features in the dataset
hiddenlayer_neurons = 3  # Number of neurons in the hidden layer
output_neurons = 1  # Number of neurons in the output layer

# Weight and bias initialization
wh = np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons))  # Input to hidden weights
bh = np.random.uniform(size=(1, hiddenlayer_neurons))  # Hidden layer bias
wout = np.random.uniform(size=(hiddenlayer_neurons, output_neurons))  # Hidden to output weights
bout = np.random.uniform(size=(1, output_neurons))  # Output layer bias

for i in range(epoch):
    # Forward Propagation
    hinp = np.dot(X, wh) + bh
    hlayer_act = sigmoid(hinp)
    outinp = np.dot(hlayer_act, wout) + bout
    output = sigmoid(outinp)

    # Backpropagation
    EO = y - output
    outgrad = derivatives_sigmoid(output)
    d_output = EO * outgrad
    EH = d_output.dot(wout.T)
    hiddengrad = derivatives_sigmoid(hlayer_act)
    d_hiddenlayer = EH * hiddengrad

    # Update weights and biases
    wout += hlayer_act.T.dot(d_output) * lr
    bout += np.sum(d_output, axis=0, keepdims=True) * lr
    wh += X.T.dot(d_hiddenlayer) * lr
    bh += np.sum(d_hiddenlayer, axis=0, keepdims=True) * lr

print("Input: \n", X)
print("Actual Output: \n", y)
print("Predicted Output: \n", output)
