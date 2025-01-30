import torchvision.datasets as ds
from torchvision import transforms
import time
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, confusion_matrix
import pickle
import numpy as np
from nn import *

epsilon = 1e-8
np.random.seed(112)


def categorical_cross_entropy(y_true, y_pred):
    """
    Categorical cross-entropy loss function for multi-class classification.
    """
    return -np.sum(y_true * np.log(y_pred + epsilon)) / len(y_pred)


def calculate_macro_f1_score(preds, labels):
    """
    Calculate macro F1 score.
    """
    true_labels = np.argmax(labels, axis=1) 
    pred_labels = np.argmax(preds, axis=1)
    return f1_score(true_labels, pred_labels, average='macro')


def train(network, loss, x_train, y_train, x_val, y_val, epochs, batch_size, initial_learning_rate, verbose):
    """
    Trains the network on the given input data for the given number of epochs.
    """
    print(f"Training the network with {epochs} epochs, {initial_learning_rate} learning rate, {batch_size} batch size")
    training_losses = []
    training_accuracies = []
    validation_losses = []
    validation_accuracies = []
    validation_macro_f1_scores = []

    for epoch in range(epochs):

        # linear LR schedule
        learning_rate = initial_learning_rate * (1 - epoch / epochs)

        correct_preds = 0
        total_preds = 0
        train_loss = 0
        batch_accuracy = []
    
        # Shuffle the data
        indices = np.random.permutation(len(x_train))
        x_train = x_train[indices]
        y_train = y_train[indices]

        for j in range(0, len(x_train), batch_size):
            x_batch = x_train[j:j+batch_size]
            y_batch = y_train[j:j+batch_size]

            # Forward propagation
            output = x_batch
            for layer in network:
                output = layer.forward_propagation(output)

            # Calculate loss
            batch_loss = loss(y_batch, output)
            train_loss += batch_loss

            # Calculate accuracy
            correct_preds += np.sum(np.argmax(output, axis=1) == np.argmax(y_batch, axis=1))
            total_preds += len(y_batch)
            batch_accuracy.append(correct_preds / total_preds)

            # Backward propagation
            output_gradient = y_batch
            for layer in reversed(network):
                output_gradient = layer.backward_propagation(output_gradient, learning_rate)

        # Calculate the average loss and accuracy for training
        train_loss /= (x_batch.shape[0])
        training_losses.append(train_loss)
        train_accuracy = np.mean(batch_accuracy)
        training_accuracies.append(train_accuracy)

        # Validation
        val_output = x_val
        for layer in network:
            if isinstance(layer, Dropout):
                val_output = layer.forward_propagation(val_output, train=False)
            else:
                val_output = layer.forward_propagation(val_output)

        val_loss = loss(y_val, val_output)
        validation_losses.append(val_loss)
        val_accuracy = np.sum(np.argmax(val_output, axis=1) == np.argmax(y_val, axis=1)) / len(y_val)
        validation_accuracies.append(val_accuracy)
        val_macro_f1_score = calculate_macro_f1_score(val_output, y_val)
        validation_macro_f1_scores.append(val_macro_f1_score)

        if verbose:
            print(f"Epoch: {epoch+1}, "
                  f"Train loss: {train_loss:.5f}, "
                  f"Train accuracy: {train_accuracy:.3f}, "
                  f"Val loss: {val_loss:.3f}, "
                  f"Val accuracy: {val_accuracy:.3f}, "
                  f"Val macro f1 score: {val_macro_f1_score:.3f}")

    return (training_losses, training_accuracies, validation_losses, 
            validation_accuracies, validation_macro_f1_scores, val_output, y_val)


def preprocess_data(x, y):
    """
    Preprocesses the data. Converts the labels to one-hot encoded vectors.
    """
    # Normalize the input data
    x = np.array(list(map(lambda img: img.flatten(), x))) / 255.0

    # one-hot encode the labels 
    num_classes = 10
    y = np.eye(num_classes)[y]
    return x, y


def load_data(train=True):
    """
    Loads the MNIST data from torchvision.
    """
    if train:
        train_data = ds.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        train_x = train_data.data.numpy()
        train_y = train_data.targets.numpy()
        return train_x, train_y
    else:
        test_data = ds.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())
        test_x = test_data.data.numpy()
        test_y = test_data.targets.numpy()
        return test_x, test_y


def save_model(network, filename):
    """
    Save the model in a pickle file.
    """
    with open(filename, 'wb') as file:
        pickle.dump(network, file)


def plot_graphs(training_losses, training_accuracies, validation_losses, validation_accuracies, validation_macro_f1_scores, val_output, y_val):
    sns.set(style="darkgrid")
    fig, axes = plt.subplots(1, 3, figsize=(15, 10))

    # Plot the training and validation losses
    sns.lineplot(x=range(len(training_losses)), y=training_losses, ax=axes[0], label="Training loss")
    sns.lineplot(x=range(len(validation_losses)), y=validation_losses, ax=axes[0], label="Validation loss")
    axes[0].set_xlabel("Epochs")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training and Validation Loss")

    # Plot the training and validation accuracies
    sns.lineplot(x=range(len(training_accuracies)), y=training_accuracies, ax=axes[1], label="Training accuracy")
    sns.lineplot(x=range(len(validation_accuracies)), y=validation_accuracies, ax=axes[1], label="Validation accuracy")
    axes[1].set_xlabel("Epochs")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Training and Validation Accuracy")

    # Plot the validation macro f1 scores
    sns.lineplot(x=range(len(validation_macro_f1_scores)), y=validation_macro_f1_scores, ax=axes[2], label="Validation macro f1 score")
    axes[2].set_xlabel("Epochs")
    axes[2].set_ylabel("Macro F1 Score")
    axes[2].set_title("Validation Macro F1 Score")

    plt.tight_layout()

    # Create a new figure for the confusion matrix
    fig_confusion, ax_confusion = plt.subplots(figsize=(15, 10))
    sns.heatmap(confusion_matrix(np.argmax(y_val, axis=1), np.argmax(val_output, axis=1)),
                annot=True, fmt='g', ax=ax_confusion, cmap="Greens")
    ax_confusion.set_xlabel("Predicted")
    ax_confusion.set_ylabel("Actual")
    ax_confusion.set_title("Confusion Matrix")

    plt.show()


def main():
    # load the data
    train_x, train_y = load_data(train=True)
    # preprocess the data
    train_x, train_y = preprocess_data(train_x, train_y)

    # Split into train and validation
    x_train, x_val, y_train, y_val = train_test_split(train_x, train_y, test_size=0.15)

    print(f"x_train: {x_train.shape}")
    print(f"y_train: {y_train.shape}")
    print(f"x_val: {x_val.shape}")
    print(f"y_val: {y_val.shape}")

    # define the network
    dense1 = Dense(784, 1024)
    dense2 = Dense(1024, 10)
    relu1 = ReLU()
    softmax = Softmax()
    dropout = Dropout(0.1)

    network = [
        dense1,
        relu1,
        dropout,
        dense2,
        softmax
    ]

    start = time.time()
    # train the network
    (training_losses, training_accuracies, validation_losses, 
     validation_accuracies, validation_macro_f1_scores, val_output, y_val) = train(network, 
                                                                                   categorical_cross_entropy, 
                                                                                   x_train, 
                                                                                   y_train, 
                                                                                   x_val, 
                                                                                   y_val, 
                                                                                   epochs=100, 
                                                                                   batch_size=1024, 
                                                                                   initial_learning_rate=5e-04, 
                                                                                   verbose=True)
    end = time.time()

    # Print time taken
    minutes, seconds = divmod(end - start, 60)
    print(f"Time taken: {minutes:.0f}m {seconds:.0f}s")

    # Remove dropout layers before saving
    network = [layer for layer in network if not isinstance(layer, Dropout)]

    # save the model
    save_model(network, "model_mnist.pickle")

    # plot the graphs
    plot_graphs(training_losses, training_accuracies, validation_losses, validation_accuracies, 
                validation_macro_f1_scores, val_output, y_val)


if __name__ == '__main__':
    main()
