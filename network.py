import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

# Load the covtype dataset
data = fetch_covtype()
X, y = data['data'], data['target']
num_classes = 7

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).long()
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).long()

# Create a DataLoader for training
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

# Define the neural network model
class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.fc = nn.Linear(input_dim, 128)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(128, output_dim)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.output(x)
        return x

# Initialize the model
model = Net(input_dim=54, output_dim=num_classes)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# Evaluation function to compute precision and recall
def evaluate(model, X, y):
    with torch.no_grad():
        outputs = model(X)
        _, predicted = torch.max(outputs, 1)
        precision = precision_score(y, predicted, average='weighted')
        recall = recall_score(y, predicted, average='weighted')
        return precision, recall

# Compute precision and recall before training
precision_before, recall_before = evaluate(model, X_test_tensor, y_test_tensor)
print(f"Precision (before training): {precision_before:.4f}")
print(f"Recall (before training): {recall_before:.4f}")

# Compute precision and recall after training
precision_after, recall_after = evaluate(model, X_test_tensor, y_test_tensor)
print(f"Precision (after training): {precision_after:.4f}")
print(f"Recall (after training): {recall_after:.4f}")
