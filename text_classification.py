import time
import pickle as pkl
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# load data
path = '/Users/sachin/Data/IMDB/'

train_path = path + 'IMDB_TRAIN'
fileObject1 = open(train_path, 'rb')
train_df = pkl.load(fileObject1)
fileObject1.close()

test_path = path + 'IMDB_TEST'
fileObject2 = open(test_path, 'rb')
test_df = pkl.load(fileObject2)
fileObject2.close()

embed_path = path + 'IMDB_EMBED'
fileObject3 = open(embed_path, 'rb')
embed_mat = pkl.load(fileObject3)
fileObject3.close()

token_path = path + 'IMDB_TOKEN'
fileObject4 = open(token_path, 'rb')
token2idx = pkl.load(fileObject4)
fileObject4.close()


# design class to read and preprocess data
class text_dataset(Dataset):
    def __init__(self, df, x_col, y_col):
        self.target = df[y_col].tolist()  
        self.sequence = df['tok_pad'].tolist()
        
    def __getitem__(self, i):
        return np.array(self.sequence[i]), self.target[i]
    
    def __len__(self):
        return len(self.sequence)        


def collate(batch):
    inputs = torch.LongTensor([item[0] for item in batch])
    targets = torch.FloatTensor([item[1] for item in batch])
    return inputs, targets

# load the dataset in dataloader
batch_size = 64
max_seq_len = 128
train_data = text_dataset(train_df, x_col = 'tok_pad', y_col = 'y')
train_data_loader = DataLoader(train_data, batch_size = batch_size, shuffle = True, 
                              num_workers=4, collate_fn = collate)

test_data = text_dataset(test_df, x_col = 'tok_pad', y_col = 'y')
test_data_loader = DataLoader(test_data, batch_size = batch_size, shuffle = True, 
                              num_workers=4, collate_fn = collate)


# design GRU model
class GRU_Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, embed_mat, non_trainable=True,
                gru_layers=2, bidirectional=True):
        super(GRU_Model, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embed_mat = embed_mat
        
        self.gru_layers = gru_layers
        self.bidirectional = bidirectional
        self.gru_hidden = 300
        self.fc1_size = 200
        self.fc2_size = 32
        self.output_size =1
        
        # Define the word embedding layer
        self.encoder = nn.Embedding(self.vocab_size, self.embed_dim)
        
        # Load embedding weights into the layer
        embed_weights = torch.tensor(self.embed_mat, dtype=torch.float)
        self.encoder.load_state_dict({'weight': embed_weights})
        
        if non_trainable:
            self.encoder.weight.requires_grad = False
            
        # create a bidirectional GRU layer
        self.gru = nn.GRU(self.embed_dim, self.gru_hidden, self.gru_layers, batch_first=True, dropout=0.5, 
                         bidirectional=self.bidirectional)
        
        self.batch_norm1 = nn.BatchNorm1d(self.fc1_size)
        self.batch_norm2 = nn.BatchNorm1d(self.fc2_size)
        
        if self.bidirectional:
            self.num_directions = 2
        else:
            self.num_directions = 1

        self.relu = nn.ReLU()
            
        self.fc1 = nn.Linear(self.gru_hidden * self.num_directions, self.fc1_size)
        self.dropout1 = nn.Dropout(0.10)
        self.fc2 = nn.Linear(self.fc1_size, self.fc2_size)
        self.dropout2 = nn.Dropout(0.05)
        self.fc3 = nn.Linear(self.fc2_size, self.output_size)
        
        
    def forward(self, x):
        
        #print("Input Shape : ", x.shape)
        out, hidden = self.gru(self.encoder(x))
        #print("Output Shape : ", out.shape)
        out = out[:,-1,:]
        out = self.relu(self.batch_norm1(self.fc1(out)))
        out = self.dropout1(out)
        out = self.relu(self.batch_norm2(self.fc2(out)))
        out = self.dropout2(out)
        out = self.fc3(out)
        return out


vocab_size = len(token2idx)
embed_dim = 300

# create model
model = GRU_Model(vocab_size, embed_dim, embed_mat, non_trainable=True, gru_layers=2, bidirectional=True)

optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.BCEWithLogitsLoss()

n_epochs=5
running_loss = []

for n_epi in range(n_epochs):
    print("epoch : ", n_epi+1)
    step = 0
    
    t5 = time.time()
    
    for i,data in enumerate(train_data_loader,0):
        step =step+1
        inputs, labels = data
        out = model(inputs)
        optimizer.zero_grad()
        loss = criterion(labels.view(-1,1), out.view(-1,1))
        print("Step : ", step+1, " Loss : ", loss.item())
        running_loss.append(loss.item())
        loss.backward()
        optimizer.step()
    
    t6 = time.time()
    print("Tiem Taken in Training Epoch : ", t6-t5)


plt.plot(running_loss)
plt.show()
