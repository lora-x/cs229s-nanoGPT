# config for training GPT-2 (124M) down to very nice loss of ~2.85 on 1 node of 8X A100 40GB
# launch as the following (e.g. in a screen session) and wait ~5 days:
# $ torchrun --standalone --nproc_per_node=8 train.py config/train_gpt2.py
# checkpoint
out_dir = 'out/wikitext'

# these make the total batch size be ~0.5M
# 12 batch size * 1024 block size * 5 gradaccum * 8 GPUs = 491,520

dataset = 'wikitext'
# init_from = 'gpt2'
init_from = 'gpt2-medium'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# e.g. 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
batch_size = 128
block_size = 1024
gradient_accumulation_steps = 5 * 8

# this makes total number of tokens be 300B
max_iters = 20 # original 600000
lr_decay_iters = 600000

# eval stuff
eval_interval = 10
eval_iters = 50
log_interval = 1

# weight decay
weight_decay = 1e-1


# logging
log_interval = 10
wandb_log = False # feel free to turn on
wandb_project = 'wikitext'
wandb_run_name = f"{init_from}_{dataset}_batch{batch_size}_eval_iters{eval_iters}_ft_iters{max_iters}-" + str(time.time())