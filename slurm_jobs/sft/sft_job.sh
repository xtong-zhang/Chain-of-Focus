#!/bin/bash


LOG_FOLDER=./logs

mkdir -p $LOG_FOLDER

export DISABLE_VERSION_CHECK=1

# export CUDA_HOME=/usr/local/cuda

export PYTHONPATH=./src:$PYTHONPATH

llamafactory-cli train ./configs/sft_lora-7b.yaml > "$LOG_FOLDER/train-sft-7b-lora.log" 2>&1 &

wait

echo "Finish Training!!! [$(date)]"
