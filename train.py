"""
This training script assumes running with torch.distributed, even though WORLD_SIZE might be 1 (single GPU)
"""

import os
import time
import math
import random
import pickle
from contextlib import nullcontext

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import transformers

from nano_model import GPT as nanoGPT
from model import GPT
from utils import get_ix

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
wandb_log = False # disabled by default
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
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
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
    # sprint(f"max = {max}")
    global batch_position
    # sprint(f"batch_position = {batch_position}")
    end_batch_position = batch_position + batch_size * block_size * batch_stride
    ix = torch.arange(batch_position, end_batch_position, block_size * batch_stride)
    ix = ix % max # wrap around
    batch_position = end_batch_position % max # wrap around
    # sprint(f"end_batch_position = {end_batch_position}")
    # sprint(f"new batch_position = {batch_position}")
    # sprint(f"ix: {ix}")
    return ix
    
def get_batch(split):
    
    data = train_data if split == 'train' else val_data
    dprint(f"{split} len(data) = {len(data)}")
    batch_stride = math.ceil((len(data) // (batch_size * block_size)) // 2) # needs to be small enough so that max_ is positive
    dprint(f"batch_stride = {batch_stride}")
    # batch_stride = 100 if split == 'train' else 3

    max_ = len(data) - batch_size * block_size * batch_stride
    dprint(f"max_ = {max_}")
    ix = get_ix(max_, batch_stride) # HACK deterministically replace ix = torch.randint(len(data) - block_size, (batch_size,))
    # ix = torch.tensor([54613,  7668, 33432, 32733, 37811,  8369, 17225, 44417]) # TODO to debug
    dprint(f"ix: {ix}")
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
nano_model = nanoGPT.from_pretrained("gpt2", dict(dropout=dropout)).cuda()
# model = nano_model
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
    # sprint("In estimate_loss()")
    # sprint("eval_iters = " + str(eval_iters))
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            # sprint(f"{split} split, iter k = {k}")
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            loss_value = loss.item()
            # sprint(f"{split} loss_value for one batch = {loss_value}")
            losses[k] = loss_value
        out[split] = losses.mean()
    # sprint("Finished estimate_loss()")
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
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
running_mfu = -1.0
best_val_loss = 1e9

for iter_num in range(max_iters):
    
    # determine and set the learning rate for this iteration
    lr = get_lr(iter_num) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
    losses = estimate_loss()
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
            torch.save(checkpoint, os.path.join(out_dir, 'ckpt.pt'))
    if iter_num == 0 and eval_only:
        break
    sprint("Starting training")
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
    t0 = t1
    if iter_num % log_interval == 0 and master_process:
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        mfu = model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
        running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        sprint(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
    

# nano_model = nanoGPT.from_pretrained("gpt2", dict(dropout=dropout)).cuda()
# nano_logits, loss = nano_model(x, y)
# sprint(f"\n Nano model logits:\n {nano_logits}")

# hf_model = transformers.GPT2LMHeadModel.from_pretrained("gpt2").cuda()
# hf_logits = hf_model(x).logits # [:, -1, :]
# sprint(f"\n Hugging Face model logits: {hf_logits}")