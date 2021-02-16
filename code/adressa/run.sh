CUDA_VISIBLE_DEVICES=$1 nohup python -u train.py --model_name=$2 --l_r=$3 --weight_decay=$4 --log_name="$2_adressa_$3lr_$4wd_$5_$6alpha" --alpha=$6 > ./log/$2_adressa_$3lr_$4wd_$5_$6alpha.txt 2>&1 &
