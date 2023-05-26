#! /bin/bash
. ~/.bashrc
conda activate PYGNLQ
nvidia-smi
python main.py --rank 1 \
       --model_name GNNLR \
       --optimizer Adam \
       --lr 0.001 \
       --dataset 5GiftCard \
       --metric ndcg@5,precision@1 \
       --test_sample_n 50 \
       --max_his 5 \
       --sparse_his 0 \
       --neg_his 1 \
       --l2 1e-4 \
       --r_logic 1e-05 \
       --r_length 1e-4 \
       --random_seed 2023 \
       --gpu 0 \
       --train 1 \
       --load 0 \
