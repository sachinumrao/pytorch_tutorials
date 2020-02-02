import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, roc_auc_score

import torch 
import torch.nn as nn
import torch.nn.functional as F

# Load dataset
iris = datasets.load_iris()

x = iris.data
y = iris.target

# Scale the features
x_scaled = preprocessing.scale(x)

# Train Test Split
x_train, x_test, y_train, y_test = train_test_split(x_scaled,
                                    y, test_size=0.3,
                                    random_state=42)

# Convert data to tensors
x_tr = torch.FloatTensor(x_train)
x_te = torch.FloatTensor(x_test)

y_tr = torch.LongTensor(y_train)
y_te = torch.LongTensor(y_test)

# Model architecture
class AnnModel(nn.Module):
    def __init__(self, inp_feats=4, h1=16, h2=8, out_feats=3):
        # Define layers
        super(AnnModel, self).__init__()
        self.fc1 = nn.Linear(inp_feats, h1)
        self.fc2 = nn.Linear(h1, h2)
        self.fc3 = nn.Linear(h2, out_feats)
        
    def forward(self, x):
        out = torch.relu(self.fc1(x))
        out = torch.relu(self.fc2(out))
        out = self.fc3(out)
        return out
        
# Set the seed
seed = 42
torch.manual_seed(seed)

# Instantiate the model
model = AnnModel()

# Error criterion
criterion = nn.CrossEntropyLoss()

# Define optimizer
optim = torch.optim.Adam(model.parameters(), lr=0.01)

# Setup training loop
num_epochs = 200
train_loss = []
test_loss = []

# Track model training time
t1 = time.time()

for i in range(num_epochs):
    # Remove previous gradients
    optim.zero_grad()

    # Score the input from model
    y_pred = model(x_tr)

    # Calculate loss
    loss = criterion(y_pred, y_tr)

    # Calculate gradient
    loss.backward()

    # Apply gradient descent step
    optim.step()

    train_loss.append(loss.item())

    with torch.no_grad():
        y_eval = model(x_te)
        loss_eval = criterion(y_eval, y_te)
        test_loss.append(loss_eval.item())

    print("Epoch: ",i+1, " Train Loss: ", 
        loss.item(), " Test Loss: ", loss_eval.item())

t2 = time.time()
print("Time Taken in Training: ", t2-t1)

# Final evaluation
with torch.no_grad():
    y_train_eval = model(x_tr)
    y_test_eval = model(x_te)

# Convert torch tensors to numpy vectors
y_train_pred = y_train_eval.numpy()
y_test_pred = y_test_eval.numpy()

y_test_actual = y_te.reshape(-1,1).numpy()
y_train_actual = y_tr.reshape(-1,1).numpy()

# One hot encode the actuals
def one_hot_enoder(x):
    x = x.reshape(-1,1)
    n_unique = np.unique(x).shape[0]
    out = np.zeros((x.shape[0], n_unique))
    for row, col in enumerate(x): 
        out[row, col] = 1 
    return out

# Convert actual labels to one-hot vector
y_train_actual = one_hot_enoder(y_train_actual)
y_test_actual = one_hot_enoder(y_test_actual)


# Convert probability output to one-hot labels
y_train_thresh = np.zeros_like(y_train_pred)
y_train_thresh[np.arange(len(y_train_pred)), 
    y_train_pred.argmax(1)] = 1

y_test_thresh = np.zeros_like(y_test_pred)
y_test_thresh[np.arange(len(y_test_pred)), 
    y_test_pred.argmax(1)] = 1


# Classification Report
train_report = classification_report(y_train_actual, y_train_thresh)
test_report = classification_report(y_test_actual, y_test_thresh)

print()
print("Classification Report:")
print("Train: ", train_report)
print("Test: ", test_report)

# Accuracy
train_acc = accuracy_score(y_train_actual, y_train_thresh)
test_acc = accuracy_score(y_test_actual, y_test_thresh)

print()
print("Accuracy:")
print("Train: ", train_acc)
print("Test: ", test_acc)

# ROC_AUC
train_roc = roc_auc_score(y_train_actual, 
                y_train_pred)
test_roc = roc_auc_score(y_test_actual, 
                y_test_pred)

print()
print("ROC-AUC:")
print("Train: ", train_roc)
print("Test: ", test_roc)

# Plot training curve
plt.plot(train_loss, label="Train")
plt.plot(test_loss, label="Test")
plt.legend()
plt.show()
