import gzip
import logging
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import spacy
import pickle
import pytorch_lightning as pl
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


torch.manual_seed(420)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


class IntentPosDataset(Dataset):
    def __init__(self, data_path: str, train: bool, intent_mapping: Dict[str, int], pos_map_path: str = "../data/pos_to_idx.json", glove: Optional[Dict[str, np.array]] = None, glove_path: str = "../../data/glove.840B.300d.pkl.gz", unk_token_strategy='average'):
        with open(data_path) as f: 
            self.data = json.load(f)
        self.intent_to_idx = intent_mapping
        self.train = train
        self.glove = glove
        self.nlp = spacy.load('en_core_web_md')
        with open(pos_map_path) as f:
            self.tag_to_idx = json.load(f)  # https://stackoverflow.com/a/50517921
        self.unk_token_strategy = unk_token_strategy
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        sample = self.data[idx]
        _id = sample['id']
        text = sample['text']
        doc = self.nlp(text)
        toks = [d.text for d in doc]
        text = self.convert_to_vectors(toks, _id)
        length = len(text)
        out = {
            'id': _id,
            'text': text,
            'length': length
        }
        if self.train:
            intent = sample['intent']
            intent = self.intent_to_idx[intent]
            tags = [d.tag_ for d in doc]
            tags = [self.tag_to_idx[t] for t in tags]
            
            out['tags'] = tags
            out['intent'] = intent
        return out
        
    def convert_to_vectors(self, text, _id):
        vectors = []
        missing_idx = []
        
        for idx, tok in enumerate(text):
            try:
                vector = torch.from_numpy(self.glove[tok]).float()
            except KeyError:
                missing_idx.append(idx)
                vectors.append(torch.zeros(300))
                continue
            else:
                vectors.append(vector)
                
        if len(vectors) == len(missing_idx):
            return torch.stack(vectors)
        
        if self.unk_token_strategy == 'ignore':
            return torch.stack(vectors)
        
        elif self.unk_token_strategy == 'average':
            if missing_idx:
                vectors = self._average_tokens(vectors, missing_idx)
                
        vectors = torch.stack(vectors)
        if torch.isnan(vectors).sum() > 0:
            print('NaN in embeddings!')
            print(_id)
            raise Exception
                
        return vectors
    
    @staticmethod
    def _average_tokens(vectors: list, missing_idxs: list, window: int = 2):
        for m in missing_idxs:
            avg = vectors[max(m-window, 0): m] + vectors[m + 1: m+1+window]
            if not avg:
                avg = torch.stack(vectors)
            else:
                avg = torch.stack(avg)
            if avg.sum() == 0:
                vectors[m] = torch.zeros(300)
                continue
            avg = avg[avg.nonzero(as_tuple=True)].view(-1, avg.shape[1])
            avg = torch.mean(avg[avg.nonzero(as_tuple=True)].view(-1, avg.shape[1]), axis=0)
            vectors[m] = avg
        
        return vectors
            
        
class IntentPosDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "../ADL21-HW1/data/intent", intent_mapping: str = "../data/intents_to_idx.json", test_path: str = None, embedding_obj: Optional[Dict[str, np.array]] = None, embedding_dir: str = "../../data/glove.840B.300d.gz", batch_size: int = 32):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.test_path = test_path
        self.batch_size = batch_size
        with open(intent_mapping) as f:
            self.intent_to_idx = json.load(f)
        if embedding_obj:
            self.emb = embedding_obj
        else:
            self.emb = self._load_glove(embedding_dir)
        
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.intent_train = IntentPosDataset(
                data_path=self.data_dir.joinpath('train.json'), 
                train=True,
                intent_mapping=self.intent_to_idx, 
                glove=self.emb
            ) 
            self.intent_val = IntentPosDataset(
                data_path=self.data_dir.joinpath('eval.json'), 
                train=True,
                intent_mapping=self.intent_to_idx, 
                glove=self.emb
            ) 
            self.tag_to_idx = self.intent_train.tag_to_idx
        elif stage == "test" or stage is None:
            self.intent_test = IntentPosDataset(
                data_path=self.test_path if self.test_path else self.data_dir.joinpath('test.json'), 
                train=False,
                intent_mapping=self.intent_to_idx, 
                glove=self.emb
            ) 
        
    def train_dataloader(self):
        return DataLoader(self.intent_train, batch_size=self.batch_size, num_workers=8, pin_memory=True, collate_fn=self._collate_fn(False), shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.intent_val, batch_size=self.batch_size, num_workers=8, pin_memory=True, collate_fn=self._collate_fn(False))

    def test_dataloader(self):
        return DataLoader(self.intent_test, batch_size=self.batch_size, num_workers=8, pin_memory=True, collate_fn=self._collate_fn(True))
        
    @staticmethod
    def _collate_fn(is_test):
        def collate_fn(batch):
            out = {}
            _id = [b['id'] for b in batch]
            text = [b['text'] for b in batch]
            length = torch.LongTensor([b['length'] for b in batch])
            text = pad_sequence(text, batch_first=True)
            if not is_test:
                intent = torch.LongTensor([b['intent'] for b in batch])
                tags = [torch.LongTensor(b['tags']) for b in batch]
                tags = pad_sequence(tags, batch_first=True, padding_value=-1)
                out['intent'] = intent
                out['tags'] = tags

            out['id'] = _id
            out['text'] = text
            out['length'] = length
            out['text'] = text
            return out
        return collate_fn
        
        
    @staticmethod
    def _load_glove(fpath: str) -> Dict[str, torch.FloatTensor]:
        logger.info("Loading GloVe embeddings...")
        with gzip.open(fpath, 'rb') as f:
            emb = pickle.load(f)
        logger.info("Done!")
        return emb
        

class TaggingPosDataset(Dataset):
    def __init__(self, data_path: str, train: bool, mapping: Dict[str, int], pos_map_path: str = "../data/pos_to_idx.json", glove: Optional[Dict[str, np.array]] = None, glove_path: str = "../../data/glove.840B.300d.gz", unk_token_strategy='average'):
        with open(data_path) as f: 
            self.data = json.load(f)
        self.tag_to_idx = mapping
        with open(pos_map_path) as f:
            self.pos_to_idx = json.load(f) 
            self.pos_to_idx['UNK'] = len(self.pos_to_idx)
        self.train = train
        self.glove = glove
        self.nlp = spacy.load('en_core_web_md')
        self.unk_token_strategy = unk_token_strategy
    
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        sample = self.data[idx]
        tokens = sample['tokens']
        tokens = [' ' if t == '' else t for t in tokens]
        _id = sample['id']
        try:
            doc = spacy.tokens.Doc(vocab=self.nlp.vocab, words=tokens)
        except Exception as e:
            print(e)
            print(f'ID: {_id}')
            print(tokens)
            raise
        tokens = self.convert_to_vectors(tokens, _id)
        length = len(tokens)
        _id = [sample['id']] * length
        out = {
            'id': _id,
            'tokens': tokens,
            'length': length
        }
        if self.train:
            UNK = self.pos_to_idx['UNK']
            tags = sample['tags']
            tags = [self.tag_to_idx[t] for t in tags]
            out['tags'] = tags
            pos = [d.tag_ for d in doc]
            out['pos'] = [self.pos_to_idx.get(p, UNK) for p in pos]
            assert len(pos) == len(tokens)
            assert len(tags) == len(tokens)
        return out
        
    def convert_to_vectors(self, text, _id):
        vectors = []
        missing_idx = []
        
        for idx, tok in enumerate(text):
            try:
                vector = torch.from_numpy(self.glove[tok]).float()
            except KeyError:
#                 avg = torch.mean(torch.stack(vectors), axis=0)
                missing_idx.append(idx)
                vectors.append(torch.zeros(300))
#                 vectors.append(avg)
                continue
            else:
                vectors.append(vector)
                
        if len(vectors) == len(missing_idx):
            return torch.stack(vectors)
        
        if self.unk_token_strategy == 'ignore':
            return torch.stack(vectors)
        
        elif self.unk_token_strategy == 'average':
            if missing_idx:
                vectors = self._average_tokens(vectors, missing_idx)
                
        vectors = torch.stack(vectors)
        if torch.isnan(vectors).sum() > 0:
            print('NaN in embeddings!')
            print(_id)
            raise Exception
                
        return vectors
    
    @staticmethod
    def _average_tokens(vectors: list, missing_idxs: list, window: int = 2):
        for m in missing_idxs:
            avg = vectors[max(m-window, 0): m] + vectors[m + 1: m+1+window]
            if not avg:
                avg = torch.stack(vectors)
            else:
                avg = torch.stack(avg)
            if avg.sum() == 0:
                vectors[m] = torch.zeros(300)
                continue
            avg = avg[avg.nonzero(as_tuple=True)].view(-1, avg.shape[1])
            avg = torch.mean(avg[avg.nonzero(as_tuple=True)].view(-1, avg.shape[1]), axis=0)
            vectors[m] = avg
        
        return vectors
            
        
class TaggingPosDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "../ADL21-HW1/data/slot/", mapping: str = "../data/tags_to_idx.json", test_path: str = None, embedding_obj: Optional[Dict[str, np.array]] = None, embedding_dir: str = "../../data/glove.840B.300d.gz", batch_size: int = 32, pin_memory: bool = True):
        super().__init__()
        self.data_dir = Path(data_dir)
        self.test_path = test_path
        self.batch_size = batch_size
        self.pin_memory = pin_memory
        with open(mapping) as f:
            self.tag_to_idx = json.load(f)
        if embedding_obj:
            self.emb = embedding_obj
        else:
            self.emb = self._load_glove(embedding_dir)
        
    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.tag_train = TaggingPosDataset(
                data_path=self.data_dir.joinpath('train.json'), 
                train=True,
                mapping=self.tag_to_idx, 
                glove=self.emb
            ) 
            self.tag_val = TaggingPosDataset(
                data_path=self.data_dir.joinpath('eval.json'), 
                train=True,
                mapping=self.tag_to_idx, 
                glove=self.emb
            ) 
            self.pos_to_idx = self.tag_train.pos_to_idx
        elif stage == "test" or stage is None:
            self.tag_test = TaggingPosDataset(
                data_path=self.test_path if self.test_path else self.data_dir.joinpath('test.json'), 
                train=False,
                mapping=self.tag_to_idx, 
                glove=self.emb
            ) 
        
    def train_dataloader(self):
        return DataLoader(self.tag_train, batch_size=self.batch_size, num_workers=8, pin_memory=self.pin_memory, collate_fn=self._collate_fn(is_test=False), shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.tag_val, batch_size=self.batch_size, num_workers=8, pin_memory=self.pin_memory, collate_fn=self._collate_fn(is_test=False))

    def test_dataloader(self):
        return DataLoader(self.tag_test, batch_size=self.batch_size, num_workers=8, pin_memory=self.pin_memory, collate_fn=self._collate_fn(is_test=True))
        
    @staticmethod
    def _collate_fn(is_test):
        def collate_fn(batch):
            out = {}
            _id = [b['id'] for b in batch]
            tokens = [b['tokens'] for b in batch]
            length = torch.LongTensor([b['length'] for b in batch])
            assert all(l == len(t) for l, t in zip(length, tokens))
            tokens = pad_sequence(tokens, batch_first=True)
            if not is_test:
                tags = [torch.LongTensor(b['tags']) for b in batch]
                pos = [torch.LongTensor(b['pos']) for b in batch]
                out['tags'] = pad_sequence(tags, batch_first=True, padding_value=-1)
                out['pos'] = pad_sequence(pos, batch_first=True, padding_value=-1)

            out['id'] = _id
            out['tokens'] = tokens
            out['length'] = length
            return out
        return collate_fn
        
    @staticmethod
    def _load_glove(fpath: str) -> Dict[str, torch.FloatTensor]:
        logger.info("Loading GloVe embeddings...")
        with gzip.open(fpath, 'rb') as f:
            emb = pickle.load(f)
        logger.info("Done!")
        return emb