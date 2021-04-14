from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path

import torch
import pytorch_lightning as pl

from models import IntentPosClassifier
from datasets import IntentPosDataModule
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
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/"
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint",
        default="../data/checkpoints/intent/intent_mt-epoch=14-training_loss=0.04-val_loss=0.06.ckpt",
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpus", type=bool_or_int, choices=[None, 1, 2], help="Use GPU or None to use CPU", default=1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    glove = IntentPosDataModule._load_glove(args.cache_dir.resolve().joinpath('glove.840B.300d.pkl.gz'))
    intent_pos_dm = IntentPosDataModule(embedding_obj=glove, test_path=args.test_file, batch_size=args.batch_size)
    intent2idx = intent_pos_dm.intent_to_idx

    trainer = pl.Trainer(
        gpus=args.gpus,
        checkpoint_callback=False,
    )

    model = IntentPosClassifier.load_from_checkpoint(args.ckpt_path)
    trainer.test(model, datamodule=intent_pos_dm)
    model.process_logits_and_save(int2idx=intent2idx, out_path=args.pred_file.resolve())

    



