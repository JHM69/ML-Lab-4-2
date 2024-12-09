
## Usage

1. **Train the model:**

   Run the `train.py` script to train the neural network on the EMNIST dataset.

   ```bash
   python train.py
   ```

2. **Test the model:**

   After training, you can test the model using the `test.py` script.

   ```bash
   python test.py
   ```

## Training

The training process involves the following steps:

- **Loading and Preprocessing Data:** The EMNIST dataset is loaded and preprocessed. The input images are normalized, and the labels are one-hot encoded.
- **Defining the Neural Network Architecture:** The architecture consists of multiple layers, including Dense, ReLU, Dropout, and Softmax layers.
- **Forward Propagation:** During each training iteration, the input data is passed through the network, layer by layer, to compute the output.
- **Loss Calculation:** The categorical cross-entropy loss is calculated between the predicted output and the true labels.
- **Backward Propagation:** The gradients of the loss with respect to the weights and biases are computed using backpropagation. The weights and biases are then updated using the Adam optimizer.
- **Validation:** After each epoch, the model is evaluated on a validation set to monitor its performance.

## Testing

The testing script evaluates the trained model on a separate test set and prints the test accuracy and macro F1 score. This helps in assessing the model's generalization ability on unseen data.

## Core Working Knowledge

### Neural Network Basics

A neural network is a computational model inspired by the way biological neural networks in the human brain process information. It consists of interconnected nodes (neurons) organized in layers:

- **Input Layer:** Receives the input data.
- **Hidden Layers:** Intermediate layers that perform computations and transformations on the input data.
- **Output Layer:** Produces the final output, which can be a classification or regression result.

### Forward Propagation

In forward propagation, the input data is passed through the network. Each neuron applies a linear transformation (weighted sum) followed by a non-linear activation function. The output of one layer becomes the input to the next layer.

### Backpropagation

Backpropagation is the process of updating the weights and biases of the network based on the error of the output. It involves:

1. **Calculating the Loss:** The difference between the predicted output and the actual labels.
2. **Computing Gradients:** Using the chain rule to compute the gradients of the loss with respect to each weight and bias.
3. **Updating Weights:** Adjusting the weights and biases using an optimization algorithm (e.g., Adam) to minimize the loss.

### Optimization

The Adam optimizer is used to update the weights during training. It combines the advantages of two other extensions of stochastic gradient descent: AdaGrad and RMSProp. Adam maintains a moving average of both the gradients and the squared gradients, allowing for adaptive learning rates for each parameter.

## Layers

The following layers are implemented in this project:

- **Dense Layer:** A fully connected layer that computes the weighted sum of inputs and applies an activation function.
- **ReLU Layer:** An activation layer that applies the Rectified Linear Unit (ReLU) function, which outputs the input directly if it is positive; otherwise, it outputs zero.
- **Softmax Layer:** An activation layer that converts the output logits into probabilities for multi-class classification.
- **Dropout Layer:** A regularization layer that randomly sets a fraction of the input units to zero during training to prevent overfitting.

This project serves as a foundational implementation of a neural network, providing insights into the core concepts and mechanics behind deep learning.