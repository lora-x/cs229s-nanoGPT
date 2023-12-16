import time

 # This is where the checkpoints will be saved
 # Take note to make sure saved checkpoints don't override each other, depending on what you are trying to accomplish (e.g. for pruning, may want to keep a few checkpoints around)
out_dir = 'out/shakespeare'
eval_interval = 10 # original: 10
eval_iters = 50 # original: 50 
# eval_iters = 50


dataset = 'shakespeare'
init_from = 'gpt2-medium' 
# init_from = 'gpt2'

# only save checkpoints if the validation loss improves
always_save_checkpoint = False

# the number of examples per iter:
# 1 batch_size * 32 grad_accum * 1024 tokens = 32,768 tokens/iter
# shakespeare has 301,966 tokens, so 1 epoch ~= 9.2 iters
batch_size = 4 # TODO
gradient_accumulation_steps = 40 # original: 40
max_iters = 20 # original: 200 TODO

# finetune at constant LR
learning_rate = 3e-5
decay_lr = False


# logging
log_interval = 10
wandb_log = True # feel free to turn on # TODO
wandb_project = 'shakespeare'
wandb_run_name = f"{init_from}_{dataset}_batch{batch_size}_eval_iters{eval_iters}_ft_iters{max_iters}-" + str(time.time())