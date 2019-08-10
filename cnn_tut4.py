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
    batch_size=32,
    shuffle=True,
    num_workers=4)

testset = torchvision.datasets.FashionMNIST(root='./data',
    train=False,
    download=True,
    transform=transform)

testloader = torch.utils.data.DataLoader(testset,
    batch_size=32,
    shuffle=True,
    num_workers=4)

# define convolutional network

import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # define convolution architecture
        # convolution params: input_channel=3, output_channel=6, kernel_szie=5
        # self.conv1 = nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1)
        self.conv1 = nn.Conv2d(1,8,3)
        # batch normalization params: input_channel=6 
        self.batch_norm1 = nn.BatchNorm2d(8)
        # max-pool layer params: kernel_size=2, stride=2
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(8,16,3)
        self.batch_norm2 = nn.BatchNorm2d(16)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.droput1 = nn.Dropout(0.10)
        self.fc2 = nn.Linear(120, 60)
        self.droput2 = nn.Dropout(0.05)
        self.fc3 = nn.Linear(60,10)

    def forward(self, x):
        # pass the input through first convolutional layer
        out = self.pool(F.relu(self.conv1(x)))
        out = self.batch_norm1(out)
        # print("Conv1 Output shape : ", out.shape)
        # pass through second conv layer
        out = self.pool(F.relu(self.conv2(out)))
        out = self.batch_norm2(out)
        # print("Conv2 Output shape : ", out.shape)
        # pass through simply connected layer
        # reshape the input for linear layer
        out = out.view(-1, 16*5*5)   ## find out how to arrive on this number
        # 16*3*3 : number of output filters from last conv layer multiply by 
        # remaining output size in that conv layer
        # apply one fully connected layer and pass through relu

        #debug
        # print("Flattend Output shape : ", out.shape)

        out = F.relu(self.fc1(out))
        out = self.droput1(out)
        # print("FC1 Output shape : ", out.shape)
        out = F.relu(self.fc2(out))
        out = self.droput2(out)
        # print("FC2 Output shape : ", out.shape)
        out = F.relu(self.fc3(out))

        # debug
        # print("Final Output shape : ", out.shape)
        return out

model = Model()

# define loss and optimizer
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.005)

# train the network in epochs
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for i,data in enumerate(trainloader, 0):
        inputs, labels = data

        # reset the gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)

        # debug
        # print("Input size : ", inputs.shape)
        # print("Output size : ", outputs.shape)
        # calculate loss
        loss = criterion(outputs, labels)

        # backward pass / calculate gradients
        loss.backward()

        # take one grad step
        optimizer.step()

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
