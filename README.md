# Qwen3-vl local setup

# Setup

## Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

## Restart your terminal, then verify:
uv --version
command -v uv

## Create environment
cd /path/to/your/repo
uv sync 

## Activate venv
source .venv/bin/activate

# Download artifacts

## HF login
hf auth login

## Download dataset
mkdir -p ./data/llava-instruct-mix
uvx --from "huggingface_hub[cli]" hf download trl-lib/llava-instruct-mix --repo-type dataset --local-dir ./data/llava-instruct-mix

## Download model: Qwen/Qwen3-VL-8B-Instruct
mkdir -p ./models/Qwen3-VL-8B-Instruct
uvx --from "huggingface_hub[cli]" hf download Qwen/Qwen3-VL-8B-Instruct --local-dir ./models/Qwen3-VL-8B-Instruct

# Run scripts

## Run training
python src/train.py

## Run inference
python src/inference.py
