"""
This training script assumes running with torch.distributed, even though WORLD_SIZE might be 1 (single GPU)
"""

import os
import time
import math
import random
import pickle
import json
from pprint import pprint
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import transformers

from model import GPT

DEBUG = False

master_process = int(os.environ.get('RANK', -1)) in (0, -1) # rank 0 or not parallel

def sprint(text):
    if master_process:
        print(text)
        
def dprint(text):
    if DEBUG == True:
        sprint(text)

# =================================== START of distributed set up copied from train.py ==========================================

# -----------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
eval_interval = 1
log_interval = 1
eval_iters = 1
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'gpt2-medium' # 'scratch' or 'resume' or 'gpt2*'
# wandb logging
wandb_log = True # disabled by default
wandb_project = 'cs229s'
wandb_run_name = 'gpt2' # 'run' + str(time.time())
# data
dataset = 'wikitext'
gradient_accumulation_steps = 5 * 8 # used to simulate larger batch sizes
batch_size = 8 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 1024
# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.1 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer
learning_rate = 6e-4 # max learning rate
max_iters = 10 # total number of training iterations
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup_iters = 2000 # how many steps to warm up for
lr_decay_iters = 600000 # should be ~= max_iters per Chinchilla
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
# DDP settings
backend = 'nccl' # 'nccl', 'gloo', etc.
# system
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
compile = False # use PyTorch 2.0 to compile the model to be faster
# -----------------------------------------------------------------------------
config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open('configurator.py').read()) # overrides from command line or config file
key_info = f"{init_from}_{dataset}_batch{batch_size}_eval_iters{eval_iters}_ft_iters{max_iters}"
config = {k: globals()[k] for k in config_keys} # will be useful for logging
# -----------------------------------------------------------------------------

# various inits, derived attributes, I/O setup
init_process_group(backend=backend)
rank = int(os.environ['RANK'])
device = f'cuda:{rank}'
torch.cuda.set_device(device)
seed_offset = rank
world_size = int(os.environ['WORLD_SIZE'])
tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
sprint(f"tokens per iteration will be: {tokens_per_iter:,}")

if master_process:
    os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(1337 + seed_offset)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
val_data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')

batch_position = 0 # make get_batch() deterministic, HACK

def get_ix(max, batch_stride):
    global batch_position
    ix = torch.arange(batch_position, 
                      batch_position + batch_size * math.floor(block_size * batch_stride), 
                      math.floor(block_size * batch_stride))
    ix = ix % max # wrap around
    batch_position = ix[-1].item() % max # wrap around
    dprint(f"new batch_position = {batch_position}")
    return ix
    
def get_batch(split):
    
    data = train_data if split == 'train' else val_data
    batch_stride = len(data) / (batch_size * block_size) / 2 # needs to be small enough so that max_ is strictly positive, can be fraction
    max_ = len(data) - math.ceil(batch_size * block_size * batch_stride)
    ix = get_ix(max_, batch_stride) # HACK deterministically replace ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# =================================== END of distributed set up copied from train.py ==========================================

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
# must initialize from OpenAI GPT-2 weights
assert init_from.startswith('gpt2')
sprint(f"Initializing from OpenAI GPT-2 weights: {init_from}")
model = GPT.from_pretrained("gpt2", dict(dropout=dropout)).cuda()
# read off the created config params, so we can store them into checkpoint correctly
for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
    model_args[k] = getattr(model.config, k)
# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value

optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type)

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            loss_value = loss.item()
            losses[k] = loss_value
        out[split] = losses.mean()
    model.train()
    return out

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

# training loop

# logging
if wandb_log and master_process:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name, config=config)
    sprint(f"Logging to wandb as {wandb_run_name}")
    
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
running_mfu = -1.0
best_val_loss = 1e9
time_per_iter_sum = 0.0
max_memory_sum = 0.0

results = {
    "config": config,
    "tokens_per_iter": tokens_per_iter,
    "losses": {
        0: None,
        10: None,
        50: None,
        100: None,
    }
}

for iter_num in range(max_iters):
    
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    losses = estimate_loss()
    if master_process:
        sprint(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        if wandb_log:
            wandb.log({
                "iter": iter_num,
                "train/loss": losses['train'],
                "val/loss": losses['val'],
                "lr": lr,
                "mfu": running_mfu*100, # convert to percentage
            })

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                sprint(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{key_info}.pt'))
    if iter_num == 0 and eval_only:
        break
    if iter_num in (0, 10, 50, 100):
        results["losses"][iter_num] = {
            "train": losses["train"].item(),
            "val": losses["val"].item(),
        }
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
        loss = loss / gradient_accumulation_steps
        X, Y = get_batch('train')
    if grad_clip != 0.0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # sprint(f"\n iter_num = {iter_num}: loss {loss}. Our model logits (train) :\n {logits}")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    time_per_iter_sum += dt
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
        running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        sprint(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    
    memory = torch.cuda.max_memory_allocated(rank)
    torch.cuda.reset_peak_memory_stats(rank)
    max_memory_sum += memory
    sprint(f"memory = {memory}")

sprint(f"best_val_loss type {type(best_val_loss)}")
# save results
sprint(f"best val loss: {best_val_loss}")
results["best_val_loss"] = best_val_loss.item() if type(best_val_loss) == torch.Tensor else best_val_loss
results["average_time_per_iter"] = time_per_iter_sum / max_iters
results["max_memory_per_gpu"] = max_memory_sum / max_iters

with open (os.path.join(out_dir, f'results_{key_info}.json'), 'w') as fout:
    if master_process:
        pprint(results)
    json.dump(results, fout)