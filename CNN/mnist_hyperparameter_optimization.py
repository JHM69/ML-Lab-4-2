import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import pickle
import os
import json
from datetime import datetime
import torch.backends.cudnn as cudnn

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV

# Enable CUDA optimizations
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

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
    def __init__(self, activation_fn=F.relu, conv1_channels=32, conv2_channels=64, 
                 fc1_size=128, dropout_rate=0.5):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, conv1_channels, 3, 1, 1)
        self.conv2 = nn.Conv2d(conv1_channels, conv2_channels, 3, 1, 1)
        self.conv_bn1 = nn.BatchNorm2d(conv1_channels)
        self.conv_bn2 = nn.BatchNorm2d(conv2_channels)
        
        # Calculate size after convolutions
        self.fc1 = nn.Linear(conv2_channels * 7 * 7, fc1_size)
        self.fc2 = nn.Linear(fc1_size, 10)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.activation_fn = activation_fn
        
    def forward(self, x):
        x = self.pool(self.activation_fn(self.conv_bn1(self.conv1(x))))
        x = self.dropout(x)
        x = self.pool(self.activation_fn(self.conv_bn2(self.conv2(x))))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(self.activation_fn(self.fc1(x)))
        x = self.fc2(x)
        return x

# Define activation functions
activation_functions = {
    'relu': F.relu,
    'leaky_relu': F.leaky_relu,
    'elu': F.elu,
    'tanh': torch.tanh
}

# Extended parameter grid with layer configurations
param_grid = {
    'learning_rate': [0.001, 0.01, 0.0001],
    'batch_size': [128, 256, 512],  # Increased batch sizes for CUDA
    'activation': list(activation_functions.keys()),
    'optimizer': ['adam', 'sgd'],
    'conv1_channels': [16, 32, 64],
    'conv2_channels': [32, 64, 128],
    'fc1_size': [64, 128, 256],
    'dropout_rate': [0.0, 0.2, 0.5]
}

def create_model(params):
    model = CNN(
        activation_fn=activation_functions[params['activation']],
        conv1_channels=params['conv1_channels'],
        conv2_channels=params['conv2_channels'],
        fc1_size=params['fc1_size'],
        dropout_rate=params['dropout_rate']
    ).to(device)
    
    if params['optimizer'] == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    else:
        optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], 
                            momentum=0.9, nesterov=True)
    
    return model, optimizer

def train_and_evaluate(params):
    # Set up CUDA optimizations
    torch.cuda.empty_cache()
    
    model, optimizer = create_model(params)
    criterion = nn.CrossEntropyLoss()
    
    # Use pin_memory for faster data transfer to GPU
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], 
                            shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=params['batch_size'], 
                           shuffle=False, pin_memory=True, num_workers=4)
    
    # Training history
    history = {'train_loss': [], 'test_acc': [], 'train_time': []}
    
    scaler = torch.cuda.amp.GradScaler()  # For automatic mixed precision
    
    for epoch in range(num_epochs):
        model.train()
        start_time = time.time()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            # Automatic mixed precision training
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
        
        epoch_time = time.time() - start_time
        history['train_time'].append(epoch_time)
        # ...rest of the training code...

# Modified grid search with early stopping
best_acc = 0
patience = 3
consecutive_no_improve = 0

# Create results directory
results_dir = 'hyperparameter_results'
os.makedirs(results_dir, exist_ok=True)

# Grid search with parameter combinations
param_combinations = [
    {
        'learning_rate': lr,
        'batch_size': bs,
        'activation': act,
        'optimizer': opt,
        'conv1_channels': c1,
        'conv2_channels': c2,
        'fc1_size': fc1,
        'dropout_rate': dr
    }
    for lr in param_grid['learning_rate']
    for bs in param_grid['batch_size']
    for act in param_grid['activation']
    for opt in param_grid['optimizer']
    for c1 in param_grid['conv1_channels']
    for c2 in param_grid['conv2_channels']
    for fc1 in param_grid['fc1_size']
    for dr in param_grid['dropout_rate']
]

# Sort parameter combinations by compute efficiency
param_combinations.sort(key=lambda x: (x['batch_size'], -x['conv1_channels']))

for params in param_combinations:
    print(f"\nTrying parameters: {params}")
    acc, model, history = train_and_evaluate(params)
    
    results.append({
        'params': params,
        'accuracy': acc,
        'history': history,
        'train_time': np.mean(history['train_time'])
    })
    
    print(f"Accuracy: {acc*100:.2f}%, Avg epoch time: {np.mean(history['train_time']):.2f}s")
    
    if acc > best_acc:
        best_acc = acc
        best_params = params
        best_model = model
        consecutive_no_improve = 0
        
        torch.save({
            'model_state_dict': best_model.state_dict(),
            'params': best_params,
            'accuracy': best_acc
        }, f'{results_dir}/best_model.pth')
    else:
        consecutive_no_improve += 1
        
    if consecutive_no_improve >= patience:
        print("Early stopping triggered")
        break

# Save all results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
with open(f'{results_dir}/optimization_results_{timestamp}.json', 'w') as f:
    json.dump({
        'results': [{
            'params': r['params'],
            'accuracy': float(r['accuracy']),
            'history': {k: [float(v) for v in vals] for k, vals in r['history'].items()}
        } for r in results],
        'best_params': best_params,
        'best_accuracy': float(best_acc)
    }, f, indent=4)

print(f"\nBest Accuracy: {best_acc*100:.2f}% with parameters:")
for k, v in best_params.items():
    print(f"{k}: {v}")

# Plot results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Accuracy Distribution by Activation Function')
for act in param_grid['activation']:
    act_results = [r['accuracy'] for r in results if r['params']['activation'] == act]
    plt.boxplot(act_results, positions=[list(activation_functions.keys()).index(act)], 
                labels=[act])
plt.ylabel('Accuracy')

plt.subplot(1, 2, 2)
plt.title('Learning Curves (Best Model)')
best_history = next(r['history'] for r in results 
                   if r['params'] == best_params)
plt.plot(best_history['train_loss'], label='Training Loss')
plt.plot(best_history['test_acc'], label='Test Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.savefig(f'{results_dir}/optimization_results_{timestamp}.png')
plt.show()
