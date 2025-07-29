#!/bin/bash
LOG_DIR="./logs"
SCRIPT="./src/metric.py"
OUT_FILE="./temp"
> "$OUT_FILE"

for filepath in "$LOG_DIR"/ab*; do
    if [[ ! -f "$filepath" ]]; then
        continue
    fi
    filename=$(basename "$filepath")
    if [[ "$filename" == ir* ]]; then
        generated_result=0
    else
        generated_result=1
    fi

    echo "Running $filename (generated_result=$generated_result)" 
    python "$SCRIPT" --input "$filepath" --generated_result "$generated_result" >> "$OUT_FILE" 2>&1
done