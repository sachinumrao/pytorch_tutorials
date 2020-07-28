import os
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

class MNISTDataset(pl.LightningDataModule):

    def __init__(self):
        pass

    def train_dataloader(self):
        mnist_train = MNIST(os.getcwd(), 
                            train=True,
                            download=True,
                            transform=transforms.ToTensor())

        mnist_train = DataLoader(mnist_train, 
                                 batch_size=64,
                                 num_workers=4)
        return mnist_train

    def val_dataloader(self):
        mnist_val = MNIST(os.getcwd(),
                          train=False,
                          download=True,
                          transform=transforms.ToTensor())

        mnist_val = DataLoader(mnist_val, 
                               batch_size=32,
                               num_workers=4)
        return mnist_val

