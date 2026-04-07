#!/bin/bash


PROJECT_ROOT="$HOME/resigrid01"
SCRIPT_NAME="$PROJECT_ROOT/Scripts/binaryclassification_animate_impl.py"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/binary_animation.log"
PID_FILE="$LOG_DIR/binary_animation.pid"

mkdir -p "$LOG_DIR"

echo "Starting the binary classification animation task..."

if [ -f "$SCRIPT_NAME" ]; then
    cd "$PROJECT_ROOT" || exit 1
    nohup python "$SCRIPT_NAME" \
        --dt 0.04 \
        --epochs 5000 \
        --eta 0.01 \
        --num_features 200 \
        --num_samples 40000 \
        > "$LOG_FILE" 2>&1 &
    echo $! > "$PID_FILE"
    echo "Started with PID $(cat "$PID_FILE")"
else
    echo "Error: $SCRIPT_NAME not found!"
    exit 1
fi

echo "Process launched at $(date)"

