import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
from torchvision import transforms
from PIL import Image

# ================================
# 1. Set Device
# ================================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ================================
# 2. Define the CNN Model (Same architecture as before)
# ================================
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        
        self.fc1 = nn.Linear(64*7*7, 128)
        self.fc2 = nn.Linear(128, 10)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # (32,14,14)
        x = self.pool(F.relu(self.conv2(x)))  # (64,7,7)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# ================================
# 3. Load the Model
# ================================
model = CNN().to(device)
with open('model.pkl', 'rb') as f:
    state_dict = pickle.load(f)
model.load_state_dict(state_dict)
model.eval()

# ================================
# 4. Transform for the Test Image
# ================================
transform = transforms.Compose([
    transforms.Resize((28, 28)),  # Ensure the image is 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# ================================
# 5. Load and Process the Test Image
# ================================
# Replace 'test.png' with your actual test image filename
img = Image.open('8.png').convert('L')  # Convert to grayscale just in case
img = transform(img).unsqueeze(0).to(device)  # Add batch dimension

# ================================
# 6. Inference
# ================================
with torch.no_grad():
    outputs = model(img)
    _, predicted = torch.max(outputs, 1)

print(f"Predicted digit: {predicted.item()}")
