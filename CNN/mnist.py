import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pickle

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np

# ================================
# 1. Set Device
# ================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ================================
# 2. Hyperparameters
# ================================
batch_size = 64
learning_rate = 0.001
num_epochs = 5

# ================================
# 3. Data Loading and Preprocessing
# ================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Mean and Std for MNIST
])

train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transform, download=True)

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# ================================
# 4. Define the CNN Model
# ================================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers
        self.fc1 = nn.Linear(64*7*7, 128)  # after two conv and two pool layers, size is reduced
        self.fc2 = nn.Linear(128, 10)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # Convolution + ReLU + Pool
        x = self.pool(F.relu(self.conv1(x)))  # Output: (32, 14, 14)
        x = self.pool(F.relu(self.conv2(x)))  # Output: (64, 7, 7)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers + ReLU
        x = F.relu(self.fc1(x))
        
        # Output layer (logits)
        x = self.fc2(x)
        return x

model = CNN().to(device)

# ================================
# 5. Loss and Optimizer
# ================================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# ================================
# 6. Training the Model
# ================================
model.train()
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

# ================================
# 7. Evaluation
# ================================
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Convert to numpy arrays
all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# ================================
# 8. Metrics using sklearn
# ================================
acc = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {acc * 100:.2f}%")

print("\nClassification Report:")
print(classification_report(all_labels, all_preds))

print("Confusion Matrix:")
cm = confusion_matrix(all_labels, all_preds)
print(cm)

# ================================
# 9. Optional: Visualize some results
# ================================
# Let's visualize a few samples from test dataset with predictions
# Create a separate loader with shuffle = True just to show random samples
visual_loader = DataLoader(test_dataset, batch_size=16, shuffle=True)
data_iter = iter(visual_loader)
images, labels = next(data_iter)
images = images.to(device)
import pickle

# Save the model's state_dict using pickle
model_state = model.state_dict()
with open('model.pkl', 'wb') as f:
    pickle.dump(model_state, f)
outputs = model(images)
_, preds = torch.max(outputs, 1)

# Move images and predictions back to CPU for plotting
images = images.cpu()
preds = preds.cpu()

fig, axes = plt.subplots(4,4, figsize=(8,8))
axes = axes.flatten()
for img, label, pred, ax in zip(images, labels, preds, axes):
    img = img.squeeze().numpy()
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Label: {label}, Pred: {pred}")
    ax.axis('off')
plt.tight_layout()
plt.show()
