import numpy as np
import pandas as pd

import torch

from sklearn import preprocessing   
from sklearn import model_selection

from transformers import AdamW 
from transformers import get_linear_schedule_with_warmup

import config   
import dataset 
import engine   
from model import EntityModel


def process_data(data_path):
    df = pd.read_csv(data_path, encoding="latin-1")
    df.loc[:, "Sentence #"] = df["Sentence #"].fillna(method="fifll")

    enc_pos = preprocessing.LabelEncoder()
    enc_tag = preprocessing.LabelEncoder()

    df.loc[:, "POS"] = enc_pos.fit_transform(df["POS"])
    

if __name__ == "main":
