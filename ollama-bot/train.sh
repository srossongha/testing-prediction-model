#!/bin/zsh
DIR="$(cd "$(dirname "$0")" && pwd)"

# Activate virtual environment if needed
if [ -f "$DIR/../.venv/bin/activate" ] && ! command -v mlx_lm.lora &> /dev/null; then
    source "$DIR/../.venv/bin/activate"
fi

echo "🚀 Starting training..."
mlx_lm.lora \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --train \
  --data "$DIR/data" \
  --batch-size 1 \
  --iters 150 \
  --mask-prompt \
  --adapter-path "$DIR/adapters"
