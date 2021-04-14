from argparse import ArgumentParser, Namespace
import logging
from pathlib import Path

import torch
import pytorch_lightning as pl

from models import TaggingPosClassifier
from datasets import TaggingPosDataModule

torch.manual_seed(420)
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
        default="../data"
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint",
        default="../data/checkpoints/tagging/tagging_mt-epoch=28-training_loss_epoch=0.05-val_loss_epoch=0.11.ckpt"
    )
    parser.add_argument("--pred_file", type=Path, default="pred.tagging.csv")
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--gpus", type=bool_or_int, choices=[None, 1, 2], help="Use GPU or None to use CPU", default=None)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    glove = TaggingPosDataModule._load_glove(args.cache_dir.resolve().joinpath('glove.840B.300d.pkl.gz'))
    tagging_pos_dm = TaggingPosDataModule(embedding_obj=glove, test_path=args.test_file, batch_size=args.batch_size)
    tag2idx = tagging_pos_dm.tag_to_idx

    trainer = pl.Trainer(
        gpus=1,
        # gpus=args.gpus,
        checkpoint_callback=False,
    )

    model = TaggingPosClassifier.load_from_checkpoint(args.ckpt_path)
    trainer.test(model, datamodule=tagging_pos_dm)
    model.process_logits_and_save(tag2idx=tag2idx, out_path=args.pred_file.resolve())