from train_1805112 import load_data, preprocess_data, calculate_macro_f1_score
import pickle
import numpy as np

test_x, test_y = preprocess_data(*load_data(train=False))
print(f"test_x: {test_x.shape}, test_y: {test_y.shape}")

# Load the model
with open('model3.pickle', 'rb') as file:
    network = pickle.load(file)

# Forward propagation
test_output = test_x
for layer in network:
    test_output = layer.forward_propagation(test_output)

test_accuracy = np.mean(np.argmax(test_output, axis=1) == np.argmax(test_y, axis=1))
print(f"Test accuracy: {test_accuracy:.3f}")

test_macro_f1_score = calculate_macro_f1_score(test_output, test_y)
print(f"Test macro f1 score: {test_macro_f1_score:.3f}")