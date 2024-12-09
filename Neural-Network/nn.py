import numpy as np

def adam_optimizer(weights, gradient, learning_rate, m, v, t, beta1=0.9, beta2=0.999, epsilon=1e-8):
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * (gradient ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    weights -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    return weights, m, v

class Layer:
    """
    Base class for layers.
    """
    def __init__(self):
        self.input = None
        self.output = None

    def forward_propagation(self, input):
        """
        Forward propagation function.
        """
        raise NotImplementedError

    def backward_propagation(self, output_gradient, learning_rate):
        """
        Backward propagation function.
        """
        raise NotImplementedError
    

class Dense(Layer):
    """
    Dense layer class. No activation function here. A fully connected layer.
    """

    def __init__(self, input_size, output_size):
        super().__init__()
        # self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0 / (input_size + output_size)) # xavier initialization
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size) # he initialization
        self.biases = np.zeros((1, output_size))

        # adam optimizer values
        self.m_w = 0
        self.v_w = 0
        self.m_b = 0
        self.v_b = 0
        self.t = 0

    def forward_propagation(self, input_data):
        """
        Forward propagation function.
        """
        self.input = input_data
        self.output = np.dot(input_data, self.weights) + self.biases
        return self.output
    

    def backward_propagation(self, output_gradient, learning_rate):
        """
        Backward propagation function. Output error is the error of the later layer, aka output gradient.
        """
        weights_gradient = np.dot(self.input.T, output_gradient) / output_gradient.shape[1]
        input_gradient = np.dot(output_gradient, self.weights.T) / output_gradient.shape[1]
        bias_gradient = np.mean(output_gradient, axis=0)

        # update weights and biases
        self.t += 1
        self.weights, self.m_w, self.v_w = adam_optimizer(self.weights, weights_gradient, learning_rate, self.m_w, self.v_w, self.t)
        self.biases, self.m_b, self.v_b = adam_optimizer(self.biases, bias_gradient, learning_rate, self.m_b, self.v_b, self.t)

        return input_gradient
    

class ReLU(Layer):
    """
    ReLU activation layer class.
    """
    def __init__(self):
        super().__init__()

    def forward_propagation(self, input_data):
        """
        Forward propagation function.
        """
        self.input = input_data
        self.output = np.maximum(0, input_data)
        return self.output
    
    def backward_propagation(self, output_gradient, learning_rate):
        """
        Backward propagation function.
        """
        return output_gradient * (self.input > 0)


class Softmax(Layer):
    """
    Softmax activation layer class.
    """
    def __init__(self):
        super().__init__()

    def forward_propagation(self, input_data):
        """
        Forward propagation function.
        """
        # print(f"input_data: {input_data}")
        self.input = input_data
        temp = np.exp(input_data - np.max(input_data, axis=-1, keepdims=True))
        self.output = temp / np.sum(temp, axis=-1, keepdims=True)
        return self.output
    
    
    def backward_propagation(self, y_true, learning_rate):
        """
        Backward propagation function.
        """
        return self.output - y_true
    

    
class Dropout(Layer):
    """
    Dropout layer class. Applies dropout to the input.
    """

    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None  # Mask to store dropped-out values during training

    def forward_propagation(self, input_data, train=True):
        """
        Forward propagation function.
        """
        if train:
            self.mask = (np.random.rand(*input_data.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            return input_data * self.mask
        else:
            return input_data

    def backward_propagation(self, output_gradient, learning_rate):
        """
        Backward propagation function.
        """
        return output_gradient * self.mask