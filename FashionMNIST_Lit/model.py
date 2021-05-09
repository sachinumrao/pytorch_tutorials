import torch
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from base_model import FCModel

import config

class FMNISTModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.channels = config.N_CHANNELS
        self.width = config.WIDTH
        self.height = config.HEIGHT

        self.hidden_size = config.HIDDEN_SIZE
        self.dropout = config.DROPOUT
        self.n_classes = config.NUM_CLASSES
        
        self.learning_rate = config.LR
        
        self.model = FCModel(self.channels, self.width, self.height,
                             self.hidden_size, self.dropout,
                             self.n_classes)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model.forward(x)
        loss = F.nll_loss(logits, y)
        self.log('train_loss', loss, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.model.forward(x)
        loss = F.nll_loss(logits, y)
        # preds = torch.argmax(logits, dim=1)
        self.log("val_loss", loss, on_step=True)
        return loss
    
