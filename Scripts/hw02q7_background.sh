echo "Starting HW02 animation job..."

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

if [ -d ".venv" ]; then
    echo "Activating virtual environment..."
    source .venv/bin/activate
else
    echo "WARNING: .venv not found. Running with system python."
fi

mkdir -p logs

# Timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOGFILE="logs/hw02_run_$TIMESTAMP.log"

echo "Logging to $LOGFILE"

# Run in background using nohup
nohup python -u scripts/binaryclassification_animate_impl.py > "$LOGFILE" 2>&1 &

# Save PID
echo $! > logs/hw02.pid

echo "======================================"
echo "Process started in background"
echo "PID: $(cat logs/hw02.pid)"
echo "Log file: $LOGFILE"
echo "======================================"
echo "You can now CLOSE the terminal safely."
