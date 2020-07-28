import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from model import Net
from dataset import MNISTDataset
 
if __name__ == "__main__":
    model = Net()
    dm = MNISTDataset()

    # Create logger
    LOG = True
    if LOG:
        logger = WandbLogger(project="MNIST_Lightning_V2")
        logger.watch(model, log='all', log_freq=100)
    else:
        logger = None

    trainer = pl.Trainer(max_epochs=50,
                         logger=logger)

    trainer.fit(model,
                dm)
