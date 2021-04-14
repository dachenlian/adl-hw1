import csv
import logging
from pathlib import Path
from typing import Dict, Optional

import pytorch_lightning as pl
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import write_to_csv

torch.manual_seed(420)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class IntentPosClassifier(pl.LightningModule):
    def __init__(self, num_intent: int, num_tags: int, hidden_size: int = 512, num_layers: int = 3, bidirectional: bool = True, lr: int = 1e-4, dropout=0, loss_ratio=0.7, multitask=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.multitask = multitask
        self.loss_ratio = loss_ratio if multitask else 0.0
        self.bidirectional = 2 if bidirectional else 1
        self.lr = lr
        self.rnn = nn.GRU(
            input_size=300, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            bidirectional=bidirectional, 
            batch_first=True,
            dropout=dropout,
        )
        self.hidden_to_labels = nn.Linear(self.hidden_size * self.bidirectional, num_intent)
        self.hidden_to_tags = nn.Linear(self.hidden_size * self.bidirectional, num_tags)
        self.dropout = nn.Dropout(dropout)
        self.save_hyperparameters()
        self.test_preds = {
            'ids': [],
            'logits': []
        }
        
    def forward(self, inpt):
        samples = inpt['text']
        tags = inpt.get('tags')
        lengths = inpt['length'].to('cpu')
        batch_size = samples.shape[0]
        
        hidden = self.init_hidden(batch_size).to(samples.device)
        samples = pack_padded_sequence(samples, lengths, batch_first=True, enforce_sorted=False)
        out, hidden = self.rnn(samples, hidden)
        hidden = self.dropout(hidden)
        hidden = torch.cat([hidden[-1,...], hidden[-2,...]], dim=1)  # concat last hidden states of forwards and backwards
        intent_logits = self.hidden_to_labels(hidden)
        if tags is not None: 
            out, out_len = pad_packed_sequence(out, batch_first=True)
            out = self.dropout(out)
            tag_logits = self.hidden_to_tags(out).permute(0, 2, 1)
            return intent_logits, tag_logits
            
        return intent_logits
        
    def _shared_step(self, batch):
        ids = batch['id']
        intent = batch['intent']
        tags = batch['tags']
        intent_logits, tag_logits = self(batch)
        intent_loss = F.cross_entropy(intent_logits, intent)
        tag_loss = F.cross_entropy(tag_logits, tags, ignore_index=-1)
        return intent_loss, tag_loss
        
    def training_step(self, batch, batch_idx):
        intent_loss, tag_loss = self._shared_step(batch)
        if self.multitask:
            intent_weight = self.loss_ratio
            tag_weight = 1.0 - self.loss_ratio
            loss = ((intent_loss * intent_weight) + (tag_loss * tag_weight)) 
        else:
            loss = intent_loss
        self.log('training_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        intent_loss, tag_loss = self._shared_step(batch)
        self.log('val_loss', intent_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return intent_loss
        
    def test_step(self, batch, batch_idx):
        ids = batch['id']
        logits = self(batch)
        self.test_preds['ids'].extend(ids)
        self.test_preds['logits'].extend(logits)

    def process_logits_and_save(self, int2idx, out_path: Path):
        idx2int = {idx: intent for intent, idx in int2idx.items()}
        logits = self.test_preds['logits']
        ids = self.test_preds['ids']
        preds = torch.stack(logits)
        preds = preds.argmax(dim=1).tolist()
        preds = [idx2int[p] for p in preds]

        write_to_csv('intent', ids, preds, out_path)

        return preds

    def init_hidden(self, batch_size):
        # return torch.normal(mean=0, std=1, size=(self.bidirectional * self.num_layers, batch_size, self.hidden_size)).to('cuda')
        return torch.normal(mean=0, std=1, size=(self.bidirectional * self.num_layers, batch_size, self.hidden_size))
        
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


class TaggingPosClassifier(pl.LightningModule):
    def __init__(self, num_labels: int, num_pos: int, hidden_size: int = 128, num_layers: int = 3, bidirectional: bool = True, lr: int = 1e-4, dropout=0, loss_ratio=0.7, multitask=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.multitask = multitask
        self.loss_ratio = loss_ratio if multitask else 0.0
        self.bidirectional = 2 if bidirectional else 1
        self.lr = lr
        self.dropout_prob = dropout
        self.rnn = nn.GRU(
            input_size=300, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            bidirectional=bidirectional, 
            batch_first=True,
            dropout=dropout,
        )
        self.hidden_to_labels = nn.Linear(self.hidden_size * self.bidirectional, num_labels)
        self.hidden_to_pos = nn.Linear(self.hidden_size * self.bidirectional, num_pos)
        self.dropout = nn.Dropout(dropout)
        self.save_hyperparameters()
        self.test_preds = {
            'ids': [],
            'logits': [],
            'lengths': [],
        }
        self.val_preds = {
            'ids': [],
            'preds': [],
            'lengths': [],
        }
        
    def forward(self, inpt):
        samples = inpt['tokens']
        pos = inpt.get('pos')
        lengths = inpt['length'].to('cpu')
        batch_size = samples.shape[0]
        
        hidden = self.init_hidden(batch_size).to(samples.device)
        samples = pack_padded_sequence(samples, lengths, batch_first=True, enforce_sorted=False)
        out, hidden = self.rnn(samples, hidden)
        out, out_lens = pad_packed_sequence(out, batch_first=True)
        out = self.dropout(out)
        if torch.isnan(out).sum() > 0:
            print(f"NaN in out!")
            raise Exception
        tag_logits = self.hidden_to_labels(out)
        if pos is not None:
            pos_logits = self.hidden_to_pos(out)
            return tag_logits, pos_logits
        return tag_logits
        
    def _shared_step(self, batch):
        tags = batch['tags']
        pos = batch['pos']
        tag_logits, pos_logits = self(batch)
        tag_logits = tag_logits.permute(0, 2, 1)  # must move classes to second dimension
        pos_logits = pos_logits.permute(0, 2, 1)
        tag_loss = F.cross_entropy(tag_logits, tags, ignore_index=-1)
        pos_loss = F.cross_entropy(pos_logits, pos, ignore_index=-1)
        return tag_loss, pos_loss
        
    def training_step(self, batch, batch_idx):
        tag_loss, pos_loss = self._shared_step(batch)
        if self.multitask:
            tag_weight = self.loss_ratio
            pos_weight = 1.0 - self.loss_ratio
            loss = ((tag_loss * tag_weight) + (pos_loss * pos_weight))
        else:
            loss = tag_loss
        self.log('training_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        tag_loss, pos_loss = self._shared_step(batch)
        loss = tag_loss
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
        
    def test_step(self, batch, batch_idx):
        ids = batch['id']
        logits = self(batch)
        lengths = batch['length']
        self.test_preds['ids'].extend([i[0] for i in ids])
        self.test_preds['logits'].extend(logits)
        self.test_preds['lengths'].extend(lengths)

    def process_logits_and_save(self, tag2idx, out_path: Path):
        preds = []
        idx2tag = {idx: intent for intent, idx in tag2idx.items()}
        logits = self.test_preds['logits']
        ids = self.test_preds['ids']
        lengths = self.test_preds['lengths']
        for length, log in zip(lengths, logits):
            log = log[:length]
            softmaxed = F.log_softmax(log, dim=1).argmax(dim=1).tolist()
            to_tag = [idx2tag[i] for i in softmaxed]
            preds.append(to_tag)

        write_to_csv(task='tagging', ids=ids, preds=preds, out_path=out_path)
            
    def init_hidden(self, batch_size):
        # return torch.normal(mean=0, std=1, size=(self.bidirectional * self.num_layers, batch_size, self.hidden_size)).to('cuda')
        return torch.normal(mean=0, std=1, size=(self.bidirectional * self.num_layers, batch_size, self.hidden_size))
        
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)