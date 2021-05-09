import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.loggers import TensorBoardLogger
from model import FMNISTModel
from data_loader import BasicDataLoader
import config

if __name__ == "__main__":
    logger = TensorBoardLogger('./logs/', name='FashionMNIST_FC')
    dm = BasicDataLoader()
    model = FMNISTModel()
    trainer = pl.Trainer(max_epochs=config.EPOCHS, 
                         gpus=1,
                         logger=logger,
                         log_every_n_steps=100)
    
    trainer.fit(model, dm)