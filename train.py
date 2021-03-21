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


