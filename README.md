# Clicks can be Cheating: Counterfactual Recommendation for Mitigating Clickbait Issue

This is the pytorch implementation of our paper at SIGIR 2021:

> [Clicks can be Cheating: Counterfactual Recommendation for Mitigating Clickbait Issue](https://arxiv.org/abs/2009.09945)
>
> Wenjie Wang, Fuli Feng, Xiangnan He, Hanwang  Zhang, Tat-Seng Chua.

## Environment

- Anaconda 3
- python 3.7.3
- pytorch 1.4.0
- numpy 1.16.4 

## Usage

#### Parameters

- model_name: MMGCN.
- l_r: learning rate. Default: 1e-3.
- weight_decay: the hyper-parameter for weight decay. Default: 1e-3.
- gpu_id: the gpu used for training. 

Other parameter settings can be found in train.py. We keep the default setings as MMGCN.

### Training

```
python train.py --model_name=$1 --l_r=$2 --weight_decay=$3
```

or use run.sh

```
sh run.sh gpu_id model_name l_r weight_decay
```

The log file will be in the ./log/ folder.

### Inference

1. Download the checkpoints released by us from [Google drive](https://drive.google.com/drive/folders/1LJNpDtj8kinqb89Dimx0OpRylwQmIZje?usp=sharing).
2. Put the '.pth' file into the model_1 folder.
3. Run inference.py or run_inference.sh:

```
python inference.py --model_name=$2 --l_r=$3 --weight_decay=$4 --log_name="$2_tiktok_$3lr_$4wd_$5"
```

```
sh run_inference.sh gpu_id model_name l_r weight_decay log_name
```

### Examples

1. Train MMGCN on Tiktok:

```
cd ./code/tiktok
CUDA_VISIBLE_DEVICES=0 python main.py --model_name=MMGCN --l_r=1e-3 --weight_decay=1e-3
```

2. Inference MMGCN on Adressa

```
cd ./code/adressa
sh run_inference.sh 0 MMGCN 1e-3 1e-3 TIE
```

## Citation  

If you use our code, please kindly cite:

```
@inproceedings{wang2021Clicks,
  title={Clicks can be Cheating: Counterfactual Recommendationfor Mitigating Clickbait Issue},
  author={Wenjie Wang, Fuli Feng, Xiangnan He, Hanwang Zhang, and Tat-Seng Chua},
  booktitle={SIGIR},
  year={2021},
  publisher={ACM}
}
```

## Acknowledgment

Thanks to the MMGCN implementation:

- [MMGCN](https://github.com/weiyinwei/MMGCN) from Yinwei Wei. 

## License

NUS © [NExT++](https://www.nextcenter.org/)