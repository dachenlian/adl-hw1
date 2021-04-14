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
For some reason, `Pytorch Lightning` detectd the GPU at first and then eventually couldn't detect it. If models need to be trained, CPU-only *should* work but obviously it will take a lot longer...
```shell
python train_intent.py --data_dir [location of training data]
```
If you want to use CPU
```shell
python train_intent.py --data_dir [location of training data]--gpus None
```
If you want to use GPU
```shell
python train_intent.py --data_dir [location of training data] --gpus 1
```

## Slot tagging
```shell
python train_tagging.py --data_dir [location of training data]
```
If you want to use CPU
```shell
python train_tagging.py --data_dir [location of training data] --gpus None
```
If you want to use GPU
```shell
python train_tagging.py --data_dir [location of training data] --gpus 1
```