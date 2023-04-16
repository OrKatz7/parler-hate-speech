import os
import gc
import re
import ast
import sys
import copy
import json
import time
import math
import string
import pickle
import random
import joblib
import itertools
import warnings
import scipy as sp
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import StratifiedKFold, GroupKFold, KFold
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
import tokenizers
import transformers
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

# ====================================================
# Dataset
# ====================================================
def prepare_input(cfg, text):
    inputs = cfg.tokenizer.encode_plus(
        text, 
        return_tensors=None, 
        add_special_tokens=True, 
        max_length=cfg.max_len,
        pad_to_max_length=True,
        truncation=True
    )
    for k, v in inputs.items():
        inputs[k] = torch.tensor(v, dtype=torch.long)
    return inputs


class TrainDataset(Dataset):
    def __init__(self, cfg, df,BackTranslation = None,stop_BackTranslation_epcoh = 3,train_transforms=None,gpt_type = ['ada','babbage']):
        
        self.cfg = cfg
        df = df.fillna("-9999")
        self.df = df
        self.texts = df['text'].values
        self.texts_bt = []
        if BackTranslation is not None:
            self.texts_bt = self.texts_bt + [df[f'BackTranslation_{row}'].values for row in BackTranslation]
        if gpt_type is not None:
            self.texts_bt = self.texts_bt + [df[f'gpt_{row}'].values for row in gpt_type]
        if len(self.texts_bt)==0:
            self.texts_bt = None
            
        self.labels = df[cfg.target_cols].values
        self.epoch = 0
        self.stop_BackTranslation_epcoh = stop_BackTranslation_epcoh
        self.train_transforms = train_transforms
        self.lang = 'en'

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        if self.epoch <self.stop_BackTranslation_epcoh and self.texts_bt is not None and self.epoch>0:
            perm = torch.randperm(len(self.texts_bt)+1)[0]
            if perm == 0:
                texts = self.texts[item]
            else:
                texts = self.texts_bt[perm-1][item]
                if texts == "-9999":
                    texts = self.texts[item]
        else:
            texts = self.texts[item]
        if self.train_transforms is not None and self.epoch <self.stop_BackTranslation_epcoh:
            texts, _ = self.train_transforms(data=(texts, self.lang))['data']
        inputs = prepare_input(self.cfg, texts)
        label = torch.tensor(self.labels[item], dtype=torch.float)
        return inputs, label
    
    def set_epoch(self,e):
        self.epoch = e
    

def collate(inputs):
    mask_len = int(inputs["attention_mask"].sum(axis=1).max())
    for k, v in inputs.items():
        inputs[k] = inputs[k][:,:mask_len]
    return inputs

