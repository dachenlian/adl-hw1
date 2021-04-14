import csv
import gzip
import json
import logging
from pathlib import Path
import pickle
import numpy as np
from typing import Dict


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def bool_or_int(v):
    if v == 'None':
        return None
    else:
        return int(v)

        
def build_glove(fpath: str, save=True, save_path='../../data/glove.840B.300d.gz') -> Dict[str, np.array]:
    logger.info("Loading Glove embeddings...")
    glove = {}
    with open(fpath) as f:
        for line in f:
            values = line.split()
            word = ''.join(values[:-300])
            vector = np.array([float(v) for v in values[-300:]])
            glove[word] = vector
            
    logger.info("GloVe embeddings loaded.")
    if save:
        logger.info("Saving GloVe to disk.")
        with gzip.open(save_path, 'wb') as f:
            pickle.dump(glove, f)
        logger.info("Save complete.")
    return glove


def build_intent_mappings(fpath: str, save=False):
    with open(fpath) as f:
        data = json.load(f)
    
    intents = list(set([i["intent"] for i in data]))
    intents_to_idx = {intent: idx for idx, intent in enumerate(intents)}
    
    if save:
        with open('../data/intents_to_idx.json', 'w') as f:
            json.dump(intents_to_idx, f, ensure_ascii=False, indent=4)
    
    return intents_to_idx

    
def write_to_csv(task: str, ids: list, preds: list, out_path: Path):
    if task == 'intent':
        header = ['id', 'intent']
    elif task == 'tagging':
        header = ['id', 'tags']
        preds = [" ".join(p) for p in preds]

    with out_path.open('w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(zip(ids, preds))

    logger.info(f'Predictions written to {out_path}')