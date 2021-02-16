# Clicks can be Cheating: Counterfactual Recommendation for Mitigating Clickbait Issue


Here, we release the code and data for the proposed counterfactual recommendation framework.

## Environment
- Anaconda 3
- python 3.7.3
- pytorch 1.4.0
- numpy 1.16.4 

## Usage

### Training
```
python train.py --model_name=$1 --l_r=$2 --weight_decay=$3
```
or use run.sh
```
sh run.sh gpu_id model_name l_r weight_decay
```
The log file will be in the ./log/ folder.

#### Parameters
- model_name: MMGCN.
- l_r: learning rate. Default: 1e-3.
- weight_decay: the hyper-parameter for weight decay. Default: 1e-3.
- gpu_id: the gpu used for training. 

Other parameter settings can be found in train.py. We keep the default setings as MMGCN.

### Examples
1. Train MMGCN on Tiktok:
```
cd ./Tiktok
CUDA_VISIBLE_DEVICES=0 python main.py --model_name=MMGCN --l_r=1e-3 --weight_decay=1e-3
```

We will release all training logs in ./log folder. The hyperparameter settings can be found in the log file. 
The well-trained model parameters will be uploaded to drivers and shared it here. 
