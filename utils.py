import json
import logging
from typing import Dict

import torch


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

def build_glove(fpath: str) -> Dict[str, torch.FloatTensor]:
    logger.info("Loading Glove embeddings...")
    glove = {}
    with open(fpath) as f:
        for line in f:
            values = line.split()
            word = ''.join(values[:-300])
            vector = torch.FloatTensor([float(v) for v in values[-300:]])
            glove[word] = vector
            
    logger.info("Glove embeddings loaded.")
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