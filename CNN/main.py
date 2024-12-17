import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# -------------------------------
# Data Loading and Preprocessing
# -------------------------------
file_path = 'Electric_Production.csv'
data = pd.read_csv(file_path)

print(data.head())

scaler = MinMaxScaler(feature_range=(0, 1))
data['IPG2211A2N_scaled'] = scaler.fit_transform(data[['IPG2211A2N']])

def create_sequences(data, sequence_length):
    X, y = [], []
    for i in range(len(data) - sequence_length):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    return np.array(X), np.array(y)

sequence_length = 12
data_values = data['IPG2211A2N_scaled'].values
X, y = create_sequences(data_values, sequence_length)

train_size = int(0.8 * len(X))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of y_train: {y_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
print(f"Shape of y_test: {y_test.shape}")

# Reshape data for PyTorch: (batch, channels, sequence_length)
# Currently X has shape (batch, sequence_length), we need (batch, 1, sequence_length)
X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

# Convert to Torch Tensors
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)  # (batch, 1)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

# Create DataLoaders
batch_size = 32
train_dataset = TensorDataset(X_train_t, y_train_t)
test_dataset = TensorDataset(X_test_t, y_test_t)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------------------
# Model Definition (PyTorch)
# -------------------------------
class CNNForecaster(nn.Module):
    def __init__(self):
        super(CNNForecaster, self).__init__()
        # Assuming input shape: (batch, 1, sequence_length=12)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # After conv and pooling:
        # sequence_length = 12
        # After first conv: length -> 12 - 3 + 1 = 10
        # After first pool: length -> floor(10/2) = 5
        # After second conv: length -> 5 - 3 + 1 = 3
        # After second pool: length -> floor(3/2) = 1
        # So final feature size = 32 * 1 = 32
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(32, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1(x)        # (batch, 64, 10)
        x = torch.relu(x)
        x = self.pool1(x)        # (batch, 64, 5)
        
        x = self.conv2(x)        # (batch, 32, 3)
        x = torch.relu(x)
        x = self.pool2(x)        # (batch, 32, 1)
        
        x = self.flatten(x)      # (batch, 32)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)          # (batch, 1)
        return x

model = CNNForecaster()

# -------------------------------
# Training Setup
# -------------------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# Training Loop
# -------------------------------
epochs = 50
for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * batch_X.size(0)
    train_loss /= len(train_loader.dataset)

    # Validation Loss
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item() * batch_X.size(0)
    val_loss /= len(test_loader.dataset)

    if (epoch+1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

# -------------------------------
# Evaluation
# -------------------------------
model.eval()
with torch.no_grad():
    train_pred = []
    for batch_X, _ in train_loader:
        preds = model(batch_X).numpy()
        train_pred.append(preds)
    train_pred = np.concatenate(train_pred, axis=0)

    test_pred = []
    for batch_X, _ in test_loader:
        preds = model(batch_X).numpy()
        test_pred.append(preds)
    test_pred = np.concatenate(test_pred, axis=0)

# Compute MAE on test set
test_mae = np.mean(np.abs(test_pred.reshape(-1) - y_test))
test_loss = np.mean((test_pred.reshape(-1) - y_test)**2)

print(f"Test Loss: {test_loss:.4f}, Test MAE: {test_mae:.4f}")

# Inverse transform predictions
train_predictions = scaler.inverse_transform(train_pred)
test_predictions = scaler.inverse_transform(test_pred)

y_train_inv = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_inv = scaler.inverse_transform(y_test.reshape(-1, 1))

# -------------------------------
# Plotting
# -------------------------------
plt.figure(figsize=(14, 7))
plt.plot(y_train_inv, label='Actual Training Data', color='blue')
plt.plot(np.arange(sequence_length, len(train_predictions) + sequence_length), train_predictions, label='Predicted Training Data', color='orange')
plt.plot(np.arange(len(train_predictions) + sequence_length * 2, len(train_predictions) + sequence_length * 2 + len(test_predictions)), y_test_inv, label='Actual Test Data', color='green')
plt.plot(np.arange(len(train_predictions) + sequence_length * 2, len(train_predictions) + sequence_length * 2 + len(test_predictions)), test_predictions, label='Predicted Test Data', color='red')
plt.title('Training vs Test Predictions')
plt.xlabel('Time Steps')
plt.ylabel('Electric Production')
plt.legend()
plt.grid()
plt.show()
