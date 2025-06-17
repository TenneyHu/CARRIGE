#!/bin/bash

TEMPS=(0.1 0.4 0.7 1.0)

for TEMP in "${TEMPS[@]}"
do
    echo "Running with temperature=$TEMP"
    nohup python ./src/llm_baseline.py "$TEMP" > "logs/baseline_qwen2.5_${TEMP}.out" 2>&1 &
done