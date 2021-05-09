from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import FashionMNIST
import pytorch_lightning as pl

class BasicDataLoader(pl.LightningDataModule):
    def __init__(self, data_dir='~/Data/', batch_size=32):
        
        super(BasicDataLoader, self).__init__()
        
        self.data_dir = data_dir
        self.batch_size = batch_size
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]
        )
        
    def prepare_data(self):
        self.train_data = FashionMNIST(self.data_dir, 
                                       train=True, 
                                       download=True,
                                       transform=self.transform)
        self.test_data = FashionMNIST(self.data_dir, 
                                      train=False, 
                                      download=True,
                                      transform=self.transform)
        
    def train_dataloader(self):
        return DataLoader(self.train_data, 
                          batch_size=self.batch_size,
                          shuffle=True, 
                          num_workers=4)
    
    def val_dataloader(self):
        return DataLoader(self.test_data, 
                          batch_size=self.batch_size,
                          num_workers=4)