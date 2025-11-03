#!/bin/bash
# Cleanup script for stuck EC-CHECK test processes

echo "Cleaning up stuck processes..."

# Kill all pretrain_gpt processes related to EC-CHECK
pkill -9 -f "pretrain_gpt.*eccheck" 2>/dev/null
pkill -9 -f "test_eccheck" 2>/dev/null

# Kill torchrun processes
pkill -9 -f "torchrun" 2>/dev/null

# Wait a bit
sleep 2

# Kill any process using port 6000
fuser -k 6000/tcp 2>/dev/null

# Check remaining processes
REMAINING=$(pgrep -f "pretrain_gpt\|torchrun" | wc -l)
echo "Remaining processes: $REMAINING"

# Show GPU status
echo ""
echo "GPU Utilization:"
nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader

echo ""
echo "Cleanup complete. You can now run the test again."

