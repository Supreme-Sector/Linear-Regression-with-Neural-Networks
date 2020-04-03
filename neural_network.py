"""
neural_network.py

This code is based off of mnielsen's work with a couple of modifications
The original code can be found at https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/src/network.py
"""

#### Libraries
# Standard library
import random

# Third-party libraries
import numpy as np

class Network:

    def __init__(self, sizes):
        """The list ``sizes`` contains the number of neurons in the
        respective layers of the network.  For example, if the list
        was [2, 3, 1] then it would be a three-layer network, with the
        first layer containing 2 neurons, the second layer 3 neurons,
        and the third layer 1 neuron.  The biases and weights for the
        network are initialized randomly, using a Gaussian
        distribution with mean 0, and variance 1.  Note that the first
        layer is assumed to be an input layer, and by convention we
        won't set any biases for those neurons, since biases are only
        ever used in computing the outputs from later layers."""
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a):
        """Return the output of the network if ``a`` is input."""
        for b, w in zip(self.biases, self.weights):
            a = identity_func(np.dot(w, a) + b)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None):
        """Train the neural network using mini-batch stochastic
        gradient descent.  The ``training_data`` is a list of tuples
        ``(x, y)`` representing the training inputs and the desired
        outputs."""


        if not test_data:
            test_data = training_data
        n = len(training_data)

        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size]
                for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            print("Epoch {0}: {1}".format(j, self.evaluate(test_data)))

    def update_mini_batch(self, mini_batch, eta):
        """Update the network's weights and biases by applying
        gradient descent using backpropagation to a single mini batch.
        The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        is the learning rate."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        x = np.column_stack([example[0] for example in mini_batch])
        y = np.column_stack([example[1] for example in mini_batch])

        nabla_b, nabla_w = self.backprop(x, y)

        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]

        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        """Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient for the cost function C_x.  ``nabla_b`` and
        ``nabla_w`` are layer-by-layer lists of numpy arrays, similar
        to ``self.biases`` and ``self.weights``."""
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # feedforward
        activation = x
        activations = [x] # list to store all the activations, layer by layer
        zs = [] # list to store all the z vectors, layer by layer
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = identity_func(z)
            activations.append(activation)


        # backward pass
        deltas = self.cost_derivative(activations[-1], y) * identity_prime(zs[-1])

        nabla_b[-1] = deltas.sum(axis=1).reshape((len(deltas), 1))
        nabla_w[-1] = np.dot(deltas, activations[-2].transpose())

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = identity_prime(z)
            deltas = np.dot(self.weights[-l+1].transpose(), deltas) * sp
            nabla_b[-l] = deltas.sum(axis=1).reshape((len(deltas), 1))
            nabla_w[-l] = np.dot(deltas, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data):
        """Computes the coefficient of determination"""
        sum_sq_total = 0
        sum_sq_res = 0
        y_avg = sum([y for (_, y) in test_data])/len(test_data)
        for (x, y) in test_data:
            sum_sq_total += np.linalg.norm(y_avg-y)
            sum_sq_res += np.linalg.norm(self.feedforward(x)-y)
        return 1-sum_sq_res/sum_sq_total

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y)

#### Miscellaneous functions
def identity_func(z):
    return z

def identity_prime(z):
    return np.ones(z.shape)
