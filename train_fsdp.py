import os
import time
import math
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group, ProcessGroup
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import torch.distributed as dist
from torch.distributed.fsdp.api import ShardingStrategy, MixedPrecision
from torch.profiler import profile, record_function, ProfilerActivity
from torch.distributed.distributed_c10d import _get_default_group
from transformers.trainer_utils import set_seed

from model import GPTConfig, GPT

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
init_from = 'gpt2-medium' # 'scratch' or 'resume' or 'gpt2*'
# data
dataset = 'wikitext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 8 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 128
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 600000 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# FSDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------
# various inits, derived attributes, I/O setup
init_process_group(backend=backend)
fsdp_rank = int(os.environ['RANK'])
fsdp_local_rank = int(os.environ['LOCAL_RANK'])
fsdp_world_size = int(os.environ['WORLD_SIZE'])
print("rank, local rank, world size", fsdp_rank, fsdp_local_rank, fsdp_world_size) # DEBUG
device = f'cuda:{fsdp_local_rank}'
torch.cuda.set_device(device)
master_process = fsdp_rank == 0 # this process will do logging, checkpointing etc.
seed_offset = fsdp_rank # each process gets a different seed
set_seed(1337 + seed_offset)
assert gradient_accumulation_steps % fsdp_world_size == 0
gradient_accumulation_steps //= fsdp_world_size
tokens_per_iter = gradient_accumulation_steps * fsdp_world_size * batch_size * block_size
print(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# ------------------------------DATA SETUP------------------------------------
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# ----------------------------MODEL and OPTIMIZER SETUP------------------------------------
# model
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
override_args = dict(dropout=dropout)
model = GPT.from_pretrained(init_from, override_args)
# read off the created config params, so we can store them into checkpoint correctly
for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
# wrap model with FSDP

model = FSDP(model, 
            mixed_precision=MixedPrecision(param_dtype=ptdtype),
            # sharding_strategy=ShardingStrategy.FULL_SHARD, 
            # process_group=(_get_default_group(),_get_default_group()),
            device_id=torch.cuda.current_device())
# optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(beta1, beta2))

# ----------------------------FSDP TRAIN------------------------------------
print(f"Begin FSDP train of local rank {fsdp_local_rank}")
# learning rate decay scheduler (cosine with warmup)
def get_lr(it):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# hyperparameters
iter_num = 0
best_val_loss = 1e9

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
model.train()
while True:
    fsdp_loss = torch.zeros(1).to(fsdp_local_rank)

    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    # train
    for micro_step in range(gradient_accumulation_steps):
        # optimizer.zero_grad()
        logits, loss = model(X, Y)
        batch_len = len(X)
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # loss backward
        loss.backward()
        fsdp_loss[0] += loss.item() / gradient_accumulation_steps
    dist.all_reduce(fsdp_loss, op=dist.ReduceOp.SUM)

    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if master_process:
        print(f"iter {iter_num}: loss {fsdp_loss[0] / fsdp_world_size:.4f}, time {dt*1000:.2f}ms, memory usage {torch.cuda.max_memory_allocated(device=device)/1e6:.2f} MB")
    iter_num += 1
    
    # termination conditions
    if iter_num > max_iters:
        break

dist.barrier()
destroy_process_group()