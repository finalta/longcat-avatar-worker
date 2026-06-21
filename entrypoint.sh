#!/bin/bash
set -e

echo "🐱 LongCat-Video-Avatar 1.5 Worker starting..."
echo "   Checkpoint dir : ${CHECKPOINT_DIR:-/runpod-volume/weights/LongCat-Video-Avatar-1.5}"
echo "   Work dir       : /LongCat-Video"

# Verify weights are present
WEIGHTS="${CHECKPOINT_DIR:-/runpod-volume/weights/LongCat-Video-Avatar-1.5}"
if [ ! -d "$WEIGHTS" ]; then
    echo "❌ ERROR: Weights not found at $WEIGHTS"
    echo "   Mount a Network Volume and download weights first:"
    echo "   huggingface-cli download meituan-longcat/LongCat-Video-Avatar-1.5 --local-dir $WEIGHTS"
    exit 1
fi

echo "✅ Weights found at $WEIGHTS"
echo "🚀 Starting RunPod handler..."

exec python -u /handler.py
