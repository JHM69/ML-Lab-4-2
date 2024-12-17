from train import load_data, preprocess_data, calculate_macro_f1_score
import pickle
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt  # Add this import for plotting
import string  # Add this import for string manipulation

test_x, test_y = load_data(train=False)
test_x, test_y = preprocess_data(test_x, test_y)
print(f"test_x: {test_x.shape}")
print(f"test_y: {test_y.shape}")

# load the model
with open('model.pickle', 'rb') as file:
    network = pickle.load(file)

test_output = test_x
for layer in network:
    test_output = layer.forward_propagation(test_output)

test_accuracy = np.sum(np.argmax(test_output, axis=1) == np.argmax(test_y, axis=1)) / len(test_y)

print(f"Test accuracy: {test_accuracy:.3f}") 

test_macro_f1_score = calculate_macro_f1_score(test_output, test_y)

print(f"Test macro f1 score: {test_macro_f1_score:.3f}")

# Debug: Print the raw output of the model
print(f"Raw model output: {test_output}")

def preprocess_image(image_path):
    """
    Load and preprocess the image for prediction.
    """
    image = Image.open(image_path).convert('L')  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to match input size (28x28 for EMNIST)
    image_array = np.array(image) / 255.0  # Normalize the image
    image_array = image_array.flatten()  # Flatten the image
    return image_array[np.newaxis, :]  # Add batch dimension


def display_image(image_array):
    """
    Display the image from the given array.
    """
    plt.imshow(image_array.reshape(28, 28), cmap='gray')  # Reshape to 28x28 for display
    plt.axis('off')  # Hide axes
    plt.show()  # Show the image

# Create a mapping from class index to character
class_mapping = {i: char for i, char in enumerate(string.ascii_uppercase)}  # Maps 0-25 to A-Z

# Load the model
with open('model.pickle', 'rb') as file:
    network = pickle.load(file)

# Get predictions from a PNG file
image_path = 'j.png'  # Specify the path to your PNG file
processed_image = preprocess_image(image_path)  # Preprocess the image

test_output = processed_image
for layer in network:
    test_output = layer.forward_propagation(test_output)

# Debug: Print the raw output of the model before softmax
print(f"Raw model output before softmax: {test_output}")

predicted_class_index = np.argmax(test_output, axis=1)[0]
predicted_character = class_mapping[predicted_class_index]  # Get the corresponding character
print(f"Predicted character: {predicted_character}")  # Print the predicted character

# Display the digit image
display_image(processed_image[0])  # Display the processed image

# If you want to calculate the macro F1 score, you can do so if you have true labels
# test_macro_f1_score = calculate_macro_f1_score(test_output, true_labels)
# print(f"Test macro f1 score: {test_macro_f1_score:.3f}")