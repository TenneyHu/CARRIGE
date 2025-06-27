#!/bin/bash

TEMPS=(0.4 0.7 1.0)
MINPS=("0.05")
MODELS=("gemma2:9b" "qwen2.5:latest" "llama3.1:latest")

for MODEL in "${MODELS[@]}"; do
  MODEL_NAME=$(echo "$MODEL" | tr ':' '_')  
  for TEMP in "${TEMPS[@]}"; do
    for MINP in "${MINPS[@]}"; do
      echo "Running with model=$MODEL, temperature=$TEMP, min_p=$MINP"
      LOG_NAME="logs/baseline_${MODEL_NAME}_temp${TEMP}_minp${MINP}.out"
      CMD="python ./src/llm_baseline.py --model $MODEL --t $TEMP --min_p $MINP"
      echo "Command: $CMD"
      nohup $CMD > "$LOG_NAME" 2>&1 &
    done
  done
done