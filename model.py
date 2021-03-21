import gzip
import logging
import json
from pathlib import Path
from typing import Dict, Optional

import spacy
import pickle
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch
from torch.optim import Adam
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


torch.manual_seed(420)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class IntentClassifier(pl.LightningModule):
    def __init__(self, num_labels, hidden_size=128, num_layers=3, bidirectional=True):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = 2 if bidirectional else 1
        self.lr = 1e-3
        self.rnn = nn.GRU(
            input_size=300, 
            hidden_size=hidden_size, 
            num_layers=num_layers, 
            bidirectional=bidirectional, 
            batch_first=True
        )
        self.hidden_to_labels = nn.Linear(hidden_size * num_layers * bidirectional, num_labels)
        self.save_hyperparameters()
        
    def forward(self, input):
        batch_size = input.shape[0]
        hidden = torch.normal(mean=0, std=1, size=(batch_size, self.bidirectional * num_layers, self.hidden_size))
        out, hidden = self.rnn(input, hidden)
        logits = self.hidden_to_labels(hidden.view(batch_size, -1))
        return logits
        
    def _shared_step(self, batch):
        ids = batch['id']
        text = batch['text']
        intent = batch['intent']
        logits = self(text)
        loss = F.cross_entropy(logits, intent)
        return loss
        
    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log('training_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self._shared_step(batch)
        self.log('val_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
        
    def test_step(self, batch, batch_idx):
        ids = batch['id']
        text = batch['text']
        intent = batch['intent']
        logits = self(text)
        
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=(self.lr))