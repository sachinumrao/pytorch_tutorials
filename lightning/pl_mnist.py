import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split

import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

class MNIST_Model(pl.LightningModule):

    def __init__(self):

        super(MNIST_Model, self).__init__()

        self.layer1 = nn.Linear(28*28, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 10)

        # Deifne loss function
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x):
        h = F.relu(self.layer1(x.view(x.size(0), -1)))
        h = F.relu(self.layer2(h))
        out = self.layer3(h)
        return out

    def prepare_data(self, val_size=0.20):
        # Create transformation for raw data
        transform = transforms.Compose([transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.308,))])

        # Download training and testing data
        mnist = torchvision.datasets.MNIST(root="~/Data/MINST", 
                            train=True, download=True, 
                            transform=transform)
        val_len = int(len(mnist)*val_size)
        train_len = len(mnist) - val_len
        segments = [train_len, val_len]
        self.mnist_train, self.mnist_val = random_split(mnist, segments)

        self.mnist_test = torchvision.datasets.MNIST(root="~/Data/MINST", 
                            train=False, download=True, 
                            transform=transform)

    def train_dataloader(self):
        train_dataloader = DataLoader(self.mnist_train, batch_size=64,
                            shuffle=True)
        return train_dataloader    

    def val_dataloader(self):
        val_dataloader = DataLoader(self.mnist_val, batch_size=64,
                            shuffle=False)
        return val_dataloader   

    def test_dataloader(self):
        test_dataloader = DataLoader(self.mnist_test, batch_size=64,
                            shuffle=False)
        return test_dataloader

    def configure_optimizers(self, lr=0.005):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        return optimizer

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.loss(preds, y)
        logs = {'loss': loss}
        context = {'loss': loss, 'log': logs}
        return context

    def validation_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)
        preds = out.argmax(dim=1, keepdim=True)
        acc = preds.eq(y.view_as(preds)).sum().item()
        context = {'val_step_loss': loss, 'val_step_accuracy': acc}
        return context

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_step_loss'] for x in outputs]).mean()
        logs = {'val_loss': avg_loss}
        context = {'val_loss': avg_loss, 'log': logs}
        return context

    def test_step(self, batch, batch_idx):
        x, y = batch
        out = self.forward(x)
        loss = self.loss(out, y)
        preds = out.argmax(dim=1, keepdim=True)
        acc = preds.eq(y.view_as(preds)).sum().item()
        context = {'test_loss': loss, 'test_accuracy': acc}
        return context

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        logs = {'avg_test_loss': avg_loss}
        context = {'avg_test_loss': avg_loss, 'log': logs}
        return context

if __name__ == '__main__':

    # Create logger
    wandb_logger = WandbLogger(project="MNIST_Lightning")

    # Create model
    model = MNIST_Model()
    # model.prepare_data()

    wandb_logger.watch(model, log='all', log_freq=100)

    # Create trainer object
    trainer = pl.Trainer(max_epochs=20, logger=wandb_logger, 
                early_stop_callback=True)

    # Train the Model
    trainer.fit(model)
