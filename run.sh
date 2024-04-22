#!/bin/bash

python preprocess.py -s
python retrieve.py -m vectorize -md esm2 -b 8
python retrieve.py -m retrieve -md esm2 -b 16 -k 71 -mt cosine -cpu

python main.py \
--mode meta \
--model esm2 \
--train_size 40 \
--train_batch 1 \
--eval_batch 52000 \
--lora_r 16 \
--learning_rate 1e-4 \
--epochs 100 \
--patience 15 \
--list_size 10 \
--max_iter 5 \
--retr_metric cosine\
--augment GEMME \
--meta_tasks 3 \
--meta_train_batch 16 \
--meta_eval_batch 128 \
--adapt_lr 5e-3 \
--adapt_steps 5 \
--cross_validation 4 \
--protein all \

python main.py \
--mode meta-transfer \
--model esm2 \
--train_size 40 \
--train_batch 16 \
--eval_batch 52000 \
--lora_r 16 \
--learning_rate 1e-4 \
--epochs 100 \
--patience 15 \
--list_size 10 \
--max_iter 5 \
--retr_metric cosine \
--augment GEMME \
--meta_tasks 3 \
--meta_train_batch 16 \
--meta_eval_batch 128 \
--adapt_lr 5e-3 \
--adapt_steps 5 \
--cross_validation 4 \
--protein all \

python main.py \
--mode meta-transfer \
--model esm2 \
--train_size 40 \
--train_batch 16 \
--eval_batch 52000 \
--lora_r 16 \
--learning_rate 1e-4 \
--epochs 100 \
--patience 15 \
--list_size 10 \
--max_iter 5 \
--retr_metric cosine \
--augment GEMME \
--meta_tasks 3 \
--meta_train_batch 16 \
--meta_eval_batch 128 \
--adapt_lr 5e-3 \
--adapt_steps 5 \
--cross_validation 4 \
--protein all \
--test \
