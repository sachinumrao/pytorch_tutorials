import torch
import torch.nn as nn  
import torch.optim as optim
import torch.nn.functional as F
import pytorch_lightning as pl


class Net(pl.LightningModule):

    def __init__(self):
        super(Net, self).__init__()
        self.layer1 = nn.Linear(28*28, 1024)
        self.layer2 = nn.Linear(1024, 128)
        self.layer3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        out = self.layer1(x)
        out = F.relu(out)
        out = self.layer2(out)
        out = F.relu(out)
        out = self.layer3(out)
        return out

    def training_step(self, batch, batch_idx):
        data, target = batch
        out = self.forward(data)
        loss = F.cross_entropy(out, target)
        preds = out.argmax(dim=1, keepdim=True)
      
        corrects = torch.eq(loss, target.view(-1, 1)).sum() / 1.0
        acc = torch.mean(corrects)

        result = pl.TrainResult(loss)
        result.log('train_loss', loss)
        result.log('train_acc', acc)
        return result

    def validation_step(self, batch, batch_idx):
        data, target = batch
        out = self.forward(data)
        loss = F.cross_entropy(out, target)
        preds = out.argmax(dim=1, keepdim=True)
      
        corrects = torch.eq(loss, target.view(-1, 1)).sum() / 1.0
        acc = torch.mean(corrects)

        result = pl.EvalResult(checkpoint_on=loss)
        result.log('val_loss', loss)
        result.log('val_acc', acc)
        return result

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer