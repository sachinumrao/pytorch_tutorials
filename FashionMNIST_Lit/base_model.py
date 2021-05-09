import torch
import torch.nn as nn
import torch.nn.functional as F

class FCModel(nn.Module):
    def __init__(self, channels, width, height, hidden_size, dropout, n_classes):
        super(FCModel, self).__init__()
        
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(channels*width*height, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dense3 = nn.Linear(hidden_size, n_classes)
        
    def forward(self, x):
        out = self.flatten(x)
        out = self.dense1(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.dense2(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.dense3(out)
        out = F.log_softmax(out, dim=1)
        return out
        