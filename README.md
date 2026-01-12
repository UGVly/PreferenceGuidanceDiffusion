


uv add --group metrics torchmetrics torch-fidelity clean-fid lpips open-clip-torch


uv add --group image-reward image-reward 
uv add --group hps hpsv2 hpsv3
uv add --group reward image-reward 
uv add --dev

uv sync --no-dev
uv sync





uv run --group train -- accelerate launch train.py





<!-- uv run accelerate launch train.py --arg1 ... --arg2 ... -->
