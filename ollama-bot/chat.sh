#!/bin/zsh
DIR="$(cd "$(dirname "$0")" && pwd)"

# Activate virtual environment if needed
if [ -f "$DIR/../.venv/bin/activate" ] && ! command -v mlx_lm.chat &> /dev/null; then
    source "$DIR/../.venv/bin/activate"
fi

echo "💬 Starting Asuna Bot chat..."
mlx_lm.chat \
  --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \
  --adapter-path "$DIR/adapters"
