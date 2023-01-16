import pandas as pd
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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
warnings.filterwarnings("ignore")
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
print(f"tokenizers.__version__: {tokenizers.__version__}")
print(f"transformers.__version__: {transformers.__version__}")
from transformers import AutoTokenizer, AutoModel, AutoConfig
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
# %env TOKENIZERS_PARALLELISM=true
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
import argparse
from utils import *

def get_result(oof_df):
    labels = oof_df[CFG.target_cols].values
    preds = oof_df[[f"pred_{c}" for c in CFG.target_cols]].values
    score, scores = get_score(labels, preds)
    LOGGER.info(f'Score: {score:<.4f}  Scores: {scores}')

class CFG:
    wandb=False
    competition='Hate'
    debug=False
    apex=True
    print_freq=100
    num_workers=6
    model="microsoft/deberta-v3-base"#"Narrativaai/deberta-v3-small-finetuned-hate_speech18"#"microsoft/deberta-v3-base"
    gradient_checkpointing=True
    scheduler='cosine' # ['linear', 'cosine']
    batch_scheduler=True
    num_cycles=0.5
    num_warmup_steps=0
    epochs=5
    encoder_lr=2e-5
    decoder_lr=2e-5
    min_lr=1e-6
    eps=1e-6
    betas=(0.9, 0.999)
    batch_size=12
    max_len=512
    weight_decay=0.01
    gradient_accumulation_steps=1
    max_grad_norm=1000
    target_cols=['label_mean']
    seed=42
    n_fold=4
    trn_fold=[0, 1, 2, 3]
    train=True
    pretrain_hate = False
    back_translation = False
    


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Hate speech')
    # parser.add_argument('--config', help = 'class from config.py')
    # parser.add_argument('--sax_csv_path',default= '/sise/liorrk-group/OrDanOfir/eeg/data/dataset_SAX.parquet')
    # parser.add_argument('--img_csv_path',default= '/sise/liorrk-group/OrDanOfir/eeg/data/img_train.csv')
    parser.add_argument('--model',default='microsoft/deberta-v3-base')
    parser.add_argument('--outputs_dir',default="../outputs_baseline/")
    # parser.add_argument('--pretrain',default="../outputs_baseline/")
    parser.add_argument('--pretrain_hate',action='store_true')
    parser.add_argument('--back_translation',action='store_true')
    parser.add_argument('--classification',action='store_true')
    
    args = parser.parse_args()
    CFG.model = args.model
    if CFG.model == 'microsoft/deberta-v2-xlarge':
        CFG.batch_size = 4
    CFG.back_translation = args.back_translation
    print(CFG.model)
    OUTPUT_DIR = args.outputs_dir +"/" + CFG.model.replace("/",".")+"/"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    LOGGER = get_logger(filename=OUTPUT_DIR+"/train")
    seed_everything(seed=42)
    if args.pretrain_hate:
        CFG.pretrain_hate = args.pretrain_hate
        # train2 = pd.read_csv("../parler-hate-speech/implicit_hate_multi7.csv")
        # train2[CFG.target_cols] = train2[CFG.target_cols] / train2[CFG.target_cols].max()
        train3 = pd.read_csv("../parler-hate-speech/toxigen.csv").sample(frac=0.5).reset_index(drop=True)
        train3[CFG.target_cols] = train3[CFG.target_cols] / train3[CFG.target_cols].max()
        if args.classification:
            train3[CFG.target_cols] = (train3[CFG.target_cols] > 0.5).values.astype(int)
        # train2 = pd.concat([train2,train3]).reset_index(drop=True)
        train2 = train3
        Fold = MultilabelStratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
        for n, (train_index, val_index) in enumerate(Fold.split(train2, train2[CFG.target_cols+['id']])):
            train2.loc[val_index, 'fold'] = int(n)
        train2['fold'] = train2['fold'].astype(int)
    else:
        train2 = None
        
    train = pd.read_csv('../parler-hate-speech/parler_annotated_data_bt.csv')
    train[CFG.target_cols] = train[CFG.target_cols] - train[CFG.target_cols].min()
    train[CFG.target_cols] = train[CFG.target_cols] / train[CFG.target_cols].max()
    if args.classification:
        train[CFG.target_cols] = (train[CFG.target_cols] > 0.5).values.astype(int)
    Fold = MultilabelStratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
    for n, (train_index, val_index) in enumerate(Fold.split(train, train[CFG.target_cols+['id']])):
        train.loc[val_index, 'fold'] = int(n)
    train['fold'] = train['fold'].astype(int)
    
    tokenizer = AutoTokenizer.from_pretrained(CFG.model)
    tokenizer.save_pretrained(OUTPUT_DIR+'tokenizer/')
    CFG.tokenizer = tokenizer
    lengths = []
    tk0 = tqdm(train['text'].fillna("").values, total=len(train))
    for text in tk0:
        length = len(tokenizer(text, add_special_tokens=False)['input_ids'])
        lengths.append(length)
    CFG.max_len = min(509,max(lengths)) + 3 # cls & sep & sep
    # CFG.max_len = 512
    LOGGER.info(f"max_len: {CFG.max_len}")    
    if CFG.train:
        oof_df = pd.DataFrame()
        if args.pretrain_hate:
            if not os.path.exists(OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold0_pre.pth"):
                _ = train_loop(CFG,train2, 0,LOGGER,OUTPUT_DIR,is_pre_train=True,is_cla=args.classification)
        for fold in range(CFG.n_fold):
            if fold in CFG.trn_fold:
                _oof_df = train_loop(CFG,train, fold,LOGGER,OUTPUT_DIR,is_pre_train=False,is_cla=args.classification)
                oof_df = pd.concat([oof_df, _oof_df])
                LOGGER.info(f"========== fold: {fold} result ==========")
                get_result(_oof_df)
        oof_df = oof_df.reset_index(drop=True)
        LOGGER.info(f"========== CV ==========")
        get_result(oof_df)
        oof_df.to_pickle(OUTPUT_DIR+'oof_df.pkl')
    
