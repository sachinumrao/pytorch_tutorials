# CNN with FashionMNIST data
# import dependencies
import torch
import torchvision
import torchvision.transforms as transforms

# tranform dataset images into tensors
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))])

# load dataset
trainset = torchvision.datasets.FashionMNIST(root='./data',
    train=True,
    download=True,
    transform=transform)

# create iterable data object
trainloader = torch.utils.data.DataLoader(trainset,
    batch_size=128,
    shuffle=True,
    num_workers=4)

testset = torchvision.datasets.FashionMNIST(root='./data',
    train=False,
    download=True,
    transform=transform)

testloader = torch.utils.data.DataLoader(testset,
    batch_size=128,
    shuffle=True,
    num_workers=4)

# define convolutional network

import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # define convolution architecture
    
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        # batch normalization params: input = No of output channels
        # max-pool layer params: kernel_size=2, stride=2

        # design of conv layer 1
        self.conv1 = nn.Conv2d(1,8,3) # input channels = 1 for FashionMNIST data
        self.batch_norm1 = nn.BatchNorm2d(8)
       
        self.pool = nn.MaxPool2d(2,2) # pool layer design is common

        # design of conv layer 2
        self.conv2 = nn.Conv2d(8,16,3)
        self.batch_norm2 = nn.BatchNorm2d(16)

        # design of conv layer 3
        # self.conv3 = nn.Conv2d(16, 32, 3)
        # self.batch_norm3 = nn.BatchNorm2d(32)

        # design of FC layer 1
        self.fc1 = nn.Linear(16*5*5, 120)
        self.droput1 = nn.Dropout(0.10)

        # design of FC layer 2
        self.fc2 = nn.Linear(120, 60)
        self.droput2 = nn.Dropout(0.05)

        # design of FC layer 3 : output classes = 10
        self.fc3 = nn.Linear(60,10)

    def forward(self, x):
        # pass the input through first convolutional layer1
        out = self.pool(F.relu(self.conv1(x)))
        out = self.batch_norm1(out)

        # pass the input through first convolutional layer2
        out = self.pool(F.relu(self.conv2(out)))
        out = self.batch_norm2(out)

        # pass the input through first convolutional layer1
        # out = F.relu(self.conv3(out))
        # out = self.batch_norm3(out)

        # rehspae input for fully connected layers
        out = out.view(-1, 16*5*5)   

        # pass the input through first FC layer1
        out = F.relu(self.fc1(out))
        out = self.droput1(out)

        # pass the input through first FC layer2
        out = F.relu(self.fc2(out))
        out = self.droput2(out)

        # pass the input through first FC layer3
        out = self.fc3(out)

        return out

model = Model()

# define loss and optimizer
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# train the network in epochs
loss_records = []
epochs = 20
for epoch in range(epochs):
    running_loss = 0.0
    for i,data in enumerate(trainloader, 0):
        inputs, labels = data

        # reset the gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)

        #print(outputs.shape)
        # calculate loss
        loss = criterion(outputs, labels)

        # backward pass / calculate gradients
        loss.backward()

        # take one grad step
        optimizer.step()

        # store loss 
        loss_records.append(loss.item())

        # print stats
        if (i+1)%100 == 0:
            running_loss = loss.item()
            print("Epoch : ", epoch+1, " , Step : ", i+1, " , Loss : ",running_loss)


# test the model
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct +=  (predicted == labels).sum().item()


print("Accuracy : ", correct/total)

# draw loss value during training
import matplotlib.pyplot as plt
plt.plot(loss_records)
plt.show()
