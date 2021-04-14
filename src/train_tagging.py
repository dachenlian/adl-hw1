from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path
from typing import Dict, Optional

import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
import torch
import torch.nn as nn
import torch.nn.functional as F

from datasets import TaggingPosDataModule
from models import TaggingPosClassifier
from utils import bool_or_int


torch.manual_seed(420)
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="../data/train_val/slot",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="../data/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="../data/checkpoints/tagging/",
    )

    # model
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--multitask", type=bool, default=True)
    parser.add_argument("--loss_ratio", type=float, default=0.7)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-4)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    # training
    parser.add_argument("--gpus", type=bool_or_int, choices=[None, 1, 2], help="Use GPU or None to use CPU", default=1)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)

    glove = TaggingPosDataModule._load_glove('../data/glove.840B.300d.pkl.gz')
    tagging_pos_dm = TaggingPosDataModule(embedding_obj=glove, data_dir=args.data_dir)
    tagging_pos_dm.prepare_data()
    tagging_pos_dm.setup('fit')
    tag_labels = tagging_pos_dm.tag_to_idx
    pos_labels = tagging_pos_dm.pos_to_idx

    if args.multitask:
        filename = 'tagging_mt-{epoch:02d}-{training_loss:.2f}-{val_loss:.2f}'
    else:
        filename = 'tagging-{epoch:02d}-{training_loss:.2f}-{val_loss:.2f}'

    logging_dir = Path('.').joinpath(args.ckpt_dir)
    logging_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f'Saving checkpoints to {str(logging_dir.resolve())}')
    model = TaggingPosClassifier(
        num_labels=len(tag_labels),
        num_pos=len(pos_labels),
        num_layers=2,
        hidden_size=1024,
        loss_ratio=0.75,
        dropout=0,
        multitask=args.multitask
        )
    
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename=filename,
        save_top_k=3,
        mode='min',
    )

    trainer = pl.Trainer(
        gpus=args.gpus,
        weights_summary='full',
        default_root_dir=str(logging_dir.resolve()),
        callbacks=[EarlyStopping(monitor='val_loss_epoch'), checkpoint_callback],
    )
    trainer.fit(model, datamodule=tagging_pos_dm)

