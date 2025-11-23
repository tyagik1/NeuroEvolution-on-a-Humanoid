import numpy as np

# ----------------------
# ACTIVATION FUNCTIONS
# ----------------------
def tanh(x):
    """
    Hyperbolic tangent activation function with input clipping for numerical stability.
    Clipping prevents overflow for very large or very small inputs.
    """
    x = np.clip(x, -50, 50)
    return np.tanh(x)

def inv_tanh(y):
    """
    Inverse of tanh function. Clipped slightly below 1/-1 to avoid division by zero.
    Useful for decoding neuron outputs back to pre-activation values.
    """
    y = np.clip(y, -0.999999, 0.999999)
    return 0.5 * np.log((1 + y) / (1 - y))

# ----------------------
# NEURAL NETWORK CLASS
# ----------------------
class NeuralNetwork():

    def __init__(self, inputsize, size, outputsize):
        """
        Initializes a recurrent neural network.
        
        Parameters:
        - inputsize : int, number of input neurons (from sensors)
        - size : int, number of hidden/internal neurons
        - outputsize : int, number of output neurons (to motors)
        """
        self.Size = size  # number of internal neurons
        self.InputSize = inputsize
        self.OutputSize = outputsize

        # ----------------------
        # NEURON STATES
        # ----------------------
        self.Voltage = np.zeros(size)  # neuron activation potentials
        self.TimeConstants = np.ones(size)  # how fast neurons respond to inputs
        self.Biases = np.zeros(size)  # neuron bias terms

        # ----------------------
        # CONNECTION WEIGHTS
        # ----------------------
        self.Weights = np.zeros((size, size))  # recurrent connections between neurons
        self.SensorWeights = np.zeros((inputsize, size))  # weights from inputs to neurons
        self.MotorWeights = np.zeros((size, outputsize))  # weights from neurons to outputs

        # ----------------------
        # OUTPUTS
        # ----------------------
        self.Output = np.zeros(size)  # neuron outputs after activation
        self.Input = np.zeros(size)  # weighted sum of inputs for each neuron

    # ----------------------
    # GENE DECODING
    # ----------------------
    def decodeGeneValue(self, val, low, high):
        """
        Maps a gene value to its corresponding weight/bias/time-constant range.
        Handles cases where val is within the expected range or normalized to [-1, 1].
        """
        if low - 1e-12 <= val <= high + 1e-12:
            return float(val)  # value already within range
        span = high - low
        if 0.0 <= val <= span + 1e-12:
            return low + float(val)
        if -1.0 - 1e-12 <= val <= 1.0 + 1e-12:
            return low + ((float(val) + 1.0) / 2.0) * span
        # fallback to clipping
        return float(np.clip(val, low, high))

    # ----------------------
    # SET PARAMETERS FROM GENE
    # ----------------------
    def setParams(self, gene, b):
        """
        Assigns weights, biases, and time constants to the network based on a genome.
        
        Parameters:
        - gene : list or np.array of gene values
        - b : list of [low, high] bounds for each gene
        """
        k = 0  # index in the gene array

        # Set recurrent neuron-to-neuron weights
        for i in range(self.Size):
            for j in range(self.Size):
                low, high = b[k]
                self.Weights[i, j] = self.decodeGeneValue(gene[k], low, high)
                k += 1

        # Set sensor-to-neuron weights
        for i in range(self.InputSize):
            for j in range(self.Size):
                low, high = b[k]
                self.SensorWeights[i, j] = self.decodeGeneValue(gene[k], low, high)
                k += 1

        # Set neuron-to-motor weights
        for i in range(self.Size):
            for j in range(self.OutputSize):
                low, high = b[k]
                self.MotorWeights[i, j] = self.decodeGeneValue(gene[k], low, high)
                k += 1

        # Set biases
        for i in range(self.Size):
            low, high = b[k]
            self.Biases[i] = self.decodeGeneValue(gene[k], low, high)
            k += 1

        # Set neuron time constants
        for i in range(self.Size):
            low, high = b[k]
            self.TimeConstants[i] = self.decodeGeneValue(gene[k], low, high)
            k += 1

        # Precompute inverse of time constants for efficiency
        self.invTimeConstants = 1.0 / self.TimeConstants

    # ----------------------
    # INITIALIZE NEURON STATE
    # ----------------------
    def initializeState(self, v):
        """
        Set initial voltages for neurons and compute initial outputs.
        """
        self.Voltage = v
        self.Output = tanh(self.Voltage + self.Biases)

    # ----------------------
    # STEP NETWORK FORWARD
    # ----------------------
    def step(self, dt, i):
        """
        Advances network state by one time step.
        
        Parameters:
        - dt : float, time step size
        - i : np.array, input vector from sensors
        """
        # Weighted sum of inputs from sensors
        self.Input = np.dot(self.SensorWeights.T, i)

        # Recurrent input from other neurons
        netinput = self.Input + np.dot(self.Weights.T, self.Output)

        # Update neuron voltage using continuous-time dynamics
        self.Voltage += dt * (self.invTimeConstants * (-self.Voltage + netinput))
        self.Voltage = np.clip(self.Voltage, -50, 50)  # clipping for stability

        # Compute neuron outputs with activation function
        self.Output = tanh(self.Voltage + self.Biases)

    # ----------------------
    # COMPUTE NETWORK OUTPUT
    # ----------------------
    def out(self):
        """
        Computes output of the network from current neuron activations.
        Returns:
        - np.array of motor outputs
        """
        return tanh(np.dot(self.MotorWeights.T, self.Output))
