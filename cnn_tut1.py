# import dependencies
import torch
import torchvision
import torchvision.transforms as transforms

# tranform dataset images into tensors
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))])

# load CIFAR10 dataset
trainset = torchvision.datasets.CIFAR10(root='./data',
    train=True,
    download=True,
    transform=transform)

# create iterable data object
trainloader = torch.utils.data.DataLoader(trainset,
    batch_size=32,
    shuffle=True,
    num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data',
    train=False,
    download=True,
    transform=transform)

testloader = torch.utils.data.DataLoader(testset,
    batch_size=32,
    shuffle=True,
    num_workers=4)

classes = ['plane', 'car', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck']

# define convolutional network

import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # define convolution architecture
        self.conv1 = nn.Conv2d(3,6,5)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(6,16,5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 80)
        self.fc3 = nn.Linear(80,10)

    def forward(self, x):
        # pass the input through first convolutional layer
        x = self.pool(F.relu(self.conv1(x)))
        
        # pass through second conv layer
        x = self.pool(F.relu(self.conv2(x)))

        # pass through simply connected layer
        # reshape the input for linear layer
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

model = Model()

# define loss and optimizer
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# train the network in epochs
epochs = 20
for epoch in range(epochs):
    running_loss = 0.0
    for i,data in enumerate(trainloader, 0):
        inputs, labels = data

        # reset the gradients
        optimizer.zero_grad()

        # forward pass
        outputs = model(inputs)

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







## TODO
# Add batch normalization layer
# Add droput between linear layers