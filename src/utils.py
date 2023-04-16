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

from data import *
from model import *
from losses import *
from transforms import get_train_transforms

# ====================================================
# Utils
# ====================================================
def MCRMSE(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]
        score = mean_squared_error(y_true, y_pred, squared=False) # RMSE
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores


def get_score(y_trues, y_preds):
    mcrmse_score, scores = MCRMSE(y_trues, y_preds)
    return mcrmse_score, scores

from sklearn.metrics import roc_auc_score,confusion_matrix
def get_score_cla(y_trues, y_preds):
    scores = []
    idxes = y_trues.shape[1]
    for i in range(idxes):
        y_true = y_trues[:,i]
        y_pred = y_preds[:,i]
        score =  -roc_auc_score(y_true, y_pred) # RMSE
        # cm = confusion_matrix(y_true, y_pred>0.5)
        # print(cm)
        scores.append(score)
    mcrmse_score = np.mean(scores)
    return mcrmse_score, scores
    
    # scores = roc_auc_score(y_trues, y_preds)
    # cm = confusion_matrix(y_trues, y_preds>0.5)
    # print(cm)
    # return -scores , -scores


def get_logger(filename="train"):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
# def get_kfold(CFG):
#     Fold = MultilabelStratifiedKFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)
#     for n, (train_index, val_index) in enumerate(Fold.split(train, train[CFG.target_cols+['id']])):
#     train.loc[val_index, 'fold'] = int(n)
#     train['fold'] = train['fold'].astype(int)
#     return train

# ====================================================
# Helper functions
# ====================================================
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

#
def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def train_fn(CFG,fold, train_loader, model, criterion, optimizer, epoch, scheduler, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler(enabled=CFG.apex)
    losses = AverageMeter()
    start = end = time.time()
    global_step = 0
    for step, (inputs, labels) in enumerate(train_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.cuda.amp.autocast(enabled=CFG.apex):
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            global_step += 1
            if CFG.batch_scheduler:
                scheduler.step()
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print('Epoch: [{0}][{1}/{2}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.8f}  '
                  .format(epoch+1, step, len(train_loader), 
                          remain=timeSince(start, float(step+1)/len(train_loader)),
                          loss=losses,
                          grad_norm=grad_norm,
                          lr=scheduler.get_lr()[0]))
    return losses.avg


def valid_fn(CFG,valid_loader, model, criterion, device,is_cla=False):
    losses = AverageMeter()
    model.eval()
    preds = []
    start = end = time.time()
    EMB = []
    print(len(valid_loader))
    for step, (inputs, labels) in enumerate(valid_loader):
        inputs = collate(inputs)
        for k, v in inputs.items():
            inputs[k] = v.to(device)
        labels = labels.to(device)
        batch_size = labels.size(0)
        with torch.no_grad():
            sentence_embeddings = model.feature(inputs)
            # sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
            y_preds = model(inputs)
            loss = criterion(y_preds, labels)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        losses.update(loss.item(), batch_size)
        if is_cla:
            y_preds = y_preds.sigmoid() 
        preds.append(y_preds.to('cpu').numpy())
        EMB.append(sentence_embeddings.to('cpu').numpy())
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
            print('EVAL: [{0}/{1}] '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  .format(step, len(valid_loader),
                          loss=losses,
                          remain=timeSince(start, float(step+1)/len(valid_loader))))
    predictions = np.concatenate(preds)
    EMB = np.concatenate(EMB)
    return losses.avg, predictions,EMB

# ====================================================
# train loop
# ====================================================
def train_loop(CFG,folds, fold,LOGGER,OUTPUT_DIR,is_pre_train=False,is_cla = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    LOGGER.info(f"========== fold: {fold} training ==========")

    # ====================================================
    # loader
    # ====================================================
    train_folds = folds[folds['fold'] != fold].reset_index(drop=True)
    valid_folds = folds[folds['fold'] == fold].reset_index(drop=True)
    valid_labels = valid_folds[CFG.target_cols].values
    train_labels = train_folds[CFG.target_cols].values
    BackTranslation = []#['de','es','fr'] if not is_pre_train and CFG.back_translation else None #,'fr','es','nl','no'
    LOGGER.info(f"BackTranslation {BackTranslation}")
    
    train_dataset = TrainDataset(CFG, train_folds , BackTranslation = BackTranslation , 
                                 stop_BackTranslation_epcoh = 3,train_transforms = get_train_transforms(),gpt_type = ['ada','babbage'] if CFG.gpt else None)
    
    train_dataset2 = TrainDataset(CFG, train_folds , BackTranslation = None , stop_BackTranslation_epcoh = 0)
    valid_dataset = TrainDataset(CFG, valid_folds , BackTranslation = None , stop_BackTranslation_epcoh = 0)

    train_loader = DataLoader(train_dataset,
                              batch_size=CFG.batch_size,
                              shuffle=True,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=True)
    
    train_val_loader = DataLoader(train_dataset2,
                              batch_size=CFG.batch_size,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)
        
    valid_loader = DataLoader(valid_dataset,
                              batch_size=CFG.batch_size * 2,
                              shuffle=False,
                              num_workers=CFG.num_workers, pin_memory=True, drop_last=False)

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CustomModel(CFG, config_path=None, pretrained=True,LOGGER=LOGGER)
    torch.save(model.config, OUTPUT_DIR+'config.pth')
    model.to(device)
    if not is_pre_train and os.path.exists(OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{0}_pre.pth"):
        model.load_state_dict(torch.load(OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{0}_pre.pth")['model'])
        name = OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{0}_pre.pth"
        LOGGER.info(f"load {name}")
    
    def get_optimizer_params(model, encoder_lr, decoder_lr, weight_decay=0.0):
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
        optimizer_parameters = [
            {'params': [p for n, p in model.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': weight_decay},
            {'params': [p for n, p in model.model.named_parameters() if any(nd in n for nd in no_decay)],
             'lr': encoder_lr, 'weight_decay': 0.0},
            {'params': [p for n, p in model.named_parameters() if "model" not in n],
             'lr': decoder_lr, 'weight_decay': 0.0}
        ]
        return optimizer_parameters

    optimizer_parameters = get_optimizer_params(model,
                                                encoder_lr=CFG.encoder_lr, 
                                                decoder_lr=CFG.decoder_lr,
                                                weight_decay=CFG.weight_decay)
    optimizer = AdamW(optimizer_parameters, lr=CFG.encoder_lr, eps=CFG.eps, betas=CFG.betas)
    
    # ====================================================
    # scheduler
    # ====================================================
    def get_scheduler(cfg, optimizer, num_train_steps):
        if cfg.scheduler == 'linear':
            scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps
            )
        elif cfg.scheduler == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=cfg.num_warmup_steps, num_training_steps=num_train_steps, num_cycles=cfg.num_cycles
            )
        return scheduler
    
    num_train_steps = int(len(train_folds) / CFG.batch_size * CFG.epochs)
    scheduler = get_scheduler(CFG, optimizer, num_train_steps)

    # ====================================================
    # loop
    # ====================================================
    if is_cla:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.SmoothL1Loss(reduction='mean') # RMSELoss(reduction="mean")
    
    best_score = np.inf
    epochs =  1 if is_pre_train else CFG.epochs
    for epoch in range(epochs):
        if epoch == 0:
            _, _,EMB_train_real = valid_fn(CFG,train_val_loader, model, criterion, device)
            avg_val_loss, predictions,EMB_real = valid_fn(CFG,valid_loader, model, criterion, device,is_cla=is_cla)
        start_time = time.time()

        # train
        train_loader.dataset.set_epoch(epoch)
        avg_loss = train_fn(CFG,fold, train_loader, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        
        avg_val_loss, predictions,EMB = valid_fn(CFG,valid_loader, model, criterion, device,is_cla=is_cla)
        
        # scoring
        if is_cla:
            score, scores = get_score_cla(valid_labels, predictions)
        else:
            print(valid_labels[0])
            print(predictions[0])
            score, scores = get_score(valid_labels, predictions)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}  Scores: {scores}')
        
        if best_score > score:
            _, _,EMB_train = valid_fn(CFG,train_val_loader, model, criterion, device)
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            if is_pre_train:
                name = OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_pre.pth"
            else:
                name = OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_best.pth"
            torch.save({'model': model.state_dict(),
                        'predictions': predictions,
                        'EMB_train_real':EMB_train_real,
                        'EMB_real':EMB_real,
                        'EMB':EMB,
                        'valid_labels':valid_labels,
                        'EMB_train':EMB_train,
                        'valid_folds':valid_folds,
                        'train_labels':train_labels,
                        "train_folds":train_folds},
                        name)
    name = 'pre' if is_pre_train else "best"
    predictions = torch.load(OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_{name}.pth", 
                             map_location=torch.device('cpu'))['predictions']
    EMB = torch.load(OUTPUT_DIR+f"{CFG.model.replace('/', '-')}_fold{fold}_{name}.pth", 
                             map_location=torch.device('cpu'))['EMB']
    valid_folds[[f"pred_{c}" for c in CFG.target_cols]] = predictions
    for i in range(EMB.shape[1]):
        valid_folds[f'emb{i}'] = EMB[:,i]
    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_folds


        
