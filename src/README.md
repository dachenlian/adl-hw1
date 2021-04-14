# Sample Code for Homework 1 ADL NTU 109 Spring

## Environment
```shell
# If you have conda, we recommend you to build a conda environment called "adl"
make
# otherwise
pip install -r requirements.txt
```

## Preprocessing
# To preprocess intent detection and slot tagging datasets
Run `download.sh` to download necessary data

## Intent detection
```shell
python train_intent.py
```
If you want to use GPU
```shell
python train_intent.py --gpus 1
```

## Slot tagging
```shell
python train_tagging.py
```
If you want to use GPU
```shell
python train_tagging.py --gpus 1
