import gzip
import json
import logging
from pathlib import Path
import pickle
from typing import Dict, Optional

import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader


torch.manual_seed(420)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class IntentDataset(Dataset):
    def __init__(self, data_path: str, train: bool, intent_mapping: Dict[str, int], glove: Optional[Dict[str, torch.FloatTensor]] = None, glove_path: str = "../../data/glove.840B.300d.gz", unk_token_strategy='ignore'):
        with open(data_path) as f: 
            self.data = json.load(f)
        self.intent_to_idx = intent_mapping
        self.train = train
        self.glove = glove
        self.unk_token_strategy = unk_token_strategy
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        sample = self.data[idx]
        _id = sample['id']
        text = sample['text']
        text = self.convert_to_vectors(text)
        out = {
            'id': _id,
            'text': text
        }
        if self.train:
            intent = sample['intent']
            intent = self.intent_to_idx[intent]
            out['intent'] = intent
        return out
        
    def convert_to_vectors(self, text):
        vectors = []
        if self.unk_token_strategy == 'ignore':
            for idx, tok in enumerate(text.split()):
                try:
                    vector = self.glove[tok]
                except KeyError:
                    continue
                else:
                    vectors.append(vector)
        return torch.stack(vectors, dim=0)
            
        
class IntentDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "../ADL21-HW1/data/intent", intent_mapping: str = "../data/intents_to_idx.json", embedding_obj=None, embedding_dir: str = "../../data/glove.840B.300d.gz", batch_size: int = 32):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.batch_size = batch_size
        with open(intent_mapping) as f:
            self.intent_to_idx = json.load(f)
        if embedding_obj:
            self.emb = embedding_obj
        else:
            self.emb = self._load_glove(embedding_dir)
        
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.intent_train = IntentDataset(
                data_path=self.data_dir.joinpath('train.json'), 
                train=True,
                intent_mapping=self.intent_to_idx, 
                glove=self.emb
            ) 
            self.intent_val = IntentDataset(
                data_path=self.data_dir.joinpath('eval.json'), 
                train=True,
                intent_mapping=self.intent_to_idx, 
                glove=self.emb
            ) 
        elif stage == "test" or stage is None:
            self.intent_test = IntentDataset(
                data_path=self.data_dir.joinpath('test.json'), 
                train=False,
                intent_mapping=self.intent_to_idx, 
                glove=self.emb
            ) 
        
    def train_dataloader(self):
        return DataLoader(self.intent_train, batch_size=self.batch_size, num_workers=8, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.intent_val, batch_size=self.batch_size, num_workers=8, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.intent_test, batch_size=self.batch_size, num_workers=8, pin_memory=True)
        
    @staticmethod
    def _load_glove(fpath: str) -> Dict[str, torch.FloatTensor]:
        logger.info("Loading GloVe embeddings...")
        with gzip.open(fpath, 'rb') as f:
            emb = pickle.load(f)
        logger.info("Done!")
        return emb
        