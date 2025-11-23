import numpy as np
import matplotlib.pyplot as plt

# ----------------------
# SINGLE NEURON (PERCEPTRON)
# ----------------------
class Perceptron():
    
    def __init__(self, inputs, function):
        """
        Initializes a single neuron (perceptron).

        Parameters:
        - inputs : int, number of input connections
        - function : activation function to apply to the weighted sum
        """
        self.i = inputs  # number of input connections
        self.w = np.random.random(size=inputs) * 2 - 1  # random weights initialized in [-1, 1]
        self.b = np.random.random() * 2 - 1  # random bias in [-1, 1]
        self.lr = np.random.random()  # learning rate, currently randomly set
        self.a = function  # activation function

    def forward(self, X):
        """
        Computes the forward pass for this perceptron.

        Parameters:
        - X : input vector of length self.i

        Returns:
        - output of the neuron after applying activation function
        """
        y = np.dot(self.w, X) + self.b  # weighted sum + bias
        return self.a(y)  # apply activation function


# ----------------------
# FULL NEURAL NETWORK
# ----------------------
class NeuralNet():
    def __init__(self, inputs, hidden, ouputs, function):
        """
        Initializes a feedforward neural network.

        Parameters:
        - inputs : int, number of input neurons
        - hidden : list of ints, number of neurons in each hidden layer
        - ouputs : int, number of output neurons
        - function : activation function for all neurons
        """
        self.i = inputs
        self.h = hidden
        self.o = ouputs

        self.hls = []  # list of hidden layers (each layer is a list of Perceptrons)

        # Create hidden layers
        for i in range(len(self.h)):
            hl = []
            for j in range(self.h[i]):
                if i == 0:
                    # First hidden layer connects to input layer
                    hl.append(Perceptron(self.i, function))
                else:
                    # Subsequent hidden layers connect to previous hidden layer
                    hl.append(Perceptron(self.h[i - 1], function))
            self.hls.append(hl)

        self.hos = []  # outputs of hidden layers
        # Create output layer, connecting to last hidden layer
        self.end = [Perceptron(self.h[-1], function) for i in range(self.o)]
        self.output = [0 for i in range(self.o)]  # store final outputs
        self.lr = 0.1  # global learning rate (currently unused)

    # ----------------------
    # SET NEURAL NETWORK PARAMETERS FROM GENE
    # ----------------------
    def setParams(self, gene):
        """
        Assigns all weights and biases from a flat gene array.

        Parameters:
        - gene : list or np.array, concatenated weights and biases for all neurons
        """
        i = 0  # index in gene array

        # Set hidden layer weights and biases
        for hl in self.hls:
            for neuron in hl:
                w_vals = gene[i: i + neuron.i]  # extract weights
                i += neuron.i
                neuron.w = np.array(w_vals, dtype=float)
                neuron.b = float(gene[i])  # extract bias
                i += 1

        # Set output layer weights and biases
        for neuron in self.end:
            w_vals = gene[i: i + neuron.i]
            i += neuron.i
            neuron.w = np.array(w_vals, dtype=float)
            neuron.b = float(gene[i])
            i += 1

    # ----------------------
    # FORWARD PASS THROUGH THE NETWORK
    # ----------------------
    def forward(self, X):
        """
        Performs a full forward pass through the network.

        Parameters:
        - X : input vector

        Returns:
        - output vector of the network
        """
        ho = X  # start with input vector
        self.hos = []  # reset hidden layer outputs

        # Pass through hidden layers
        for hl in self.hls:
            ho = [neuron.forward(ho) for neuron in hl]  # forward pass through each neuron
            self.hos.append(ho)  # store layer output

        # Pass through output layer
        for i in range(len(self.output)):
            self.output[i] = self.end[i].forward(self.hos[-1])  # each output neuron takes last hidden layer as input

        return self.output
