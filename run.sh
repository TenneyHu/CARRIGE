#!/bin/bash

TEMPS=(0.7)
MODELS=("llama3.1:latest")

for MODEL in "${MODELS[@]}"; do
  MODEL_NAME=$(echo "$MODEL" | tr ':' '_')  

  for TEMP in "${TEMPS[@]}"; do
    echo "Running with model=$MODEL, temperature=$TEMP"

    LOG_NAME="logs/xxxx.out"
    CMD="python ./src/carrige_infer.py --model $MODEL --t $TEMP"

    echo "Command: $CMD"
    nohup $CMD > "$LOG_NAME" 2>&1 &
  done
done