#!/bin/bash

# Path to the conda installation
CONDA_PATH="/nlp/scr/loraxie/miniconda3"

# Source the conda configuration
source "${CONDA_PATH}/etc/profile.d/conda.sh"

# Activate the desired conda environment
conda activate cs229s

# # shakespeare
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node 4 --nnodes 1 train.py config/train_shakespeare1.py
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node 4 --nnodes 1 train.py config/train_shakespeare2.py
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node 4 --nnodes 1 train.py config/train_shakespeare4.py
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node 4 --nnodes 1 train.py config/train_shakespeare8.py
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node 4 --nnodes 1 train.py config/train_shakespeare16.py
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node 4 --nnodes 1 train.py config/train_shakespeare32.py
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node 4 --nnodes 1 train.py config/train_shakespeare64.py
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node 4 --nnodes 1 train.py config/train_shakespeare128.py

# # gpt2
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node 4 --nnodes 1 train.py config/train_wikitext1-gpt2.py
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node 4 --nnodes 1 train.py config/train_wikitext2-gpt2.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node 4 --nnodes 1 train.py config/train_wikitext4-gpt2.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node 4 --nnodes 1 train.py config/train_wikitext8-gpt2.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node 4 --nnodes 1 train.py config/train_wikitext16-gpt2.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node 4 --nnodes 1 train.py config/train_wikitext32-gpt2.py
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node 4 --nnodes 1 train.py config/train_wikitext64-gpt2.py
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node 4 --nnodes 1 train.py config/train_wikitext128-gpt2.py

# # # gpt2-medium
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node 4 --nnodes 1 train.py config/train_wikitext1.py
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node 4 --nnodes 1 train.py config/train_wikitext2.py
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node 4 --nnodes 1 train.py config/train_wikitext4.py
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node 4 --nnodes 1 train.py config/train_wikitext8.py
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node 4 --nnodes 1 train.py config/train_wikitext16.py
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node 4 --nnodes 1 train.py config/train_wikitext32.py
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node 4 --nnodes 1 train.py config/train_wikitext64.py
# CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.run --standalone --nproc_per_node 4 --nnodes 1 train.py config/train_wikitext128.py
