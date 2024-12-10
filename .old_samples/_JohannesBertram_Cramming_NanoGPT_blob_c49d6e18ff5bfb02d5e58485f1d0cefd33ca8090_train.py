"""
This script is an adaptation of Andrej Karpathy's training script for nanoGPT.
The intention is to train the original nanoGPT model on a single GPU within
a specified training budget (for example 1 day). 
"""

import os
import time
import math
import pickle
from contextlib import nullcontext
import tiktoken
from transformers import BertTokenizer
import json

import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model import GPTConfig, GPT

# Parameters that I changed to test different setting

output_type = "res"
seed = 5

torch.manual_seed(seed)
sec_per_day = 79200

learning_rate = 6e-4 # max learning rate
min_lr = 6e-5 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla

min_acc = 1 # min accumuluation steps at start of batch_size schedule
max_acc = 32 # max accumulation steps at end of batch_size schedule. If equal to min_acc, no schedule used
acc_increase = 1 # between 0 and 1, fraction at which to finish increasing the batch size
acc_warmup = 0 # between 0 and acc_increase, fraction at which to start increasing the batch size
use_acc_scheduler = True

batch_size = 16 # if gradient_accumulation_steps > 1, this is the micro-batch size
block_size = 512

lr_decay = 1 # should be ~= max_iters per Chinchilla

datatype = "ci"
set_vocab_size = 30592

eval_intervals = np.append(np.arange(0, sec_per_day - 360, 720), np.arange(sec_per_day - 120, sec_per_day, 110))

optimizer_type = "AdamW" # Lion and Sophia implemented but need to change manually in code below

exp_name = f"{output_type}_{datatype}_{set_vocab_size}_{optimizer_type}_{min_acc}_{max_acc}_{acc_warmup}_{acc_increase}_{batch_size}_{block_size}_{learning_rate}_{min_lr}_{seed}"

# saving training progress
train_info = torch.zeros((6, len(eval_intervals) + 2))
torch.save(train_info, f"{exp_name}.pt")

# -----------------------------------------------------------------------------
# Parameters below that I kept the same
# ------------------------------------------------------------------------------
# default config values designed to train a gpt2 (124M) on OpenWebText
# I/O
out_dir = 'out'
log_interval = 1
eval_iters = 100
eval_only = False # if True, script exits right after the first eval
always_save_checkpoint = False # if True, always save a checkpoint after each eval
init_from = 'scratch' # 'scratch' or 'resume' or 'gpt2*'

# data
dataset = 'openwebtext'


# model
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
bias = False # do we use bias inside LayerNorm and Linear layers?
# adamw optimizer


weight_decay = 1e-1 if optimizer_type == "AdamW" else 2e-2

beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
# learning rate decay settings
decay_lr = True # whether to decay the learning rate
warmup = 1/300 # how many steps to warm up for

if optimizer_type == "Lion":
    #weight_decay = 5e-1
    learning_rate /= 5
    min_lr /= 5
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

if use_acc_scheduler:
    min_tokens_per_iter = min_acc * batch_size * block_size
    print(f"min tokens per iteration will be: {min_tokens_per_iter:,}")
    max_tokens_per_iter = max_acc * batch_size * block_size
    print(f"max tokens per iteration will be: {max_tokens_per_iter:,}")
else:
    tokens_per_iter = gradient_accumulation_steps * batch_size * block_size
    print(f"tokens per iteration will be: {tokens_per_iter:,}")

os.makedirs(out_dir, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
# note: float16 data type will automatically use a GradScaler
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

# poor man's data loader
data_dir = os.path.join('data', dataset)
def get_batch(split):
    # We recreate np.memmap every batch to avoid a memory leak, as per
    # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
    if split == 'train':
        if datatype == "ci":
            data = np.memmap(os.path.join(data_dir, 'bert_train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - 1024, (batch_size,))
    else:
        if datatype == "ci":
            data = np.memmap(os.path.join(data_dir, 'bert_val.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - 1024, (4,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# init these up here, can override if init_from='resume' (i.e. from a checkpoint)
iter_num = 0
best_val_loss = 1e9

# attempt to derive vocab_size from the dataset
meta_path = os.path.join(data_dir, 'meta.pkl')
meta_vocab_size = None
if os.path.exists(meta_path):
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    meta_vocab_size = meta['vocab_size']
    print(f"found vocab_size = {meta_vocab_size} (inside {meta_path})")

# model init
model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=block_size,
                  bias=bias, vocab_size=None, dropout=dropout) # start with model_args from command line
if init_from == 'scratch':
    # init a new model from scratch
    print("Initializing a new model from scratch")
    # determine the vocab size we'll use for from-scratch training
    
    if set_vocab_size > 0:
        model_args['vocab_size'] = set_vocab_size
        print(f"case-insensitive vocab_size of {set_vocab_size} (33323 rounded up for efficiency)")
    else:    
        if meta_vocab_size is None:
            print("defaulting to vocab_size of GPT-2 to 50304 (50257 rounded up for efficiency)")
        model_args['vocab_size'] = meta_vocab_size if meta_vocab_size is not None else 50304
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
elif init_from == 'resume':
    print(f"Resuming training from {out_dir}")
    # resume training from a checkpoint.
    ckpt_path = os.path.join(out_dir, f'ckpt_{exp_name}.pt')
    checkpoint = torch.load(ckpt_path, map_location=device)
    checkpoint_model_args = checkpoint['model_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
        model_args[k] = checkpoint_model_args[k]
    # create the model
    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    iter_num = checkpoint['iter_num']
    best_val_loss = checkpoint['best_val_loss']

# crop down the model block size if desired, using model surgery
if block_size < model.config.block_size:
    model.crop_block_size(block_size)
    model_args['block_size'] = block_size # so that the checkpoint will have the right value
model.to(device)

# initialize a GradScaler. If enabled=False scaler is a no-op
scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

# optimizer
optimizer = model.configure_optimizers(weight_decay, learning_rate, (beta1, beta2), device_type, optimizer_type)
if init_from == 'resume':
    optimizer.load_state_dict(checkpoint['optimizer'])
checkpoint = None # free up memory

# compile the model
if compile:
    print("compiling the model... (takes a ~minute)")
    unoptimized_model = model
    model = torch.compile(model) # requires PyTorch 2.0


# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        if split == 'train':
            out[split] = 0
        else:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                print("#########")
                print(X)
                print("Y ######")
                print(Y)
                with ctx:
                    logits, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out


# learning rate decay scheduler (cosine with warmup)
def get_lr_timed(tp):
    # 1) linear warmup for warmup steps
    if tp / sec_per_day < warmup:
        return learning_rate * tp / sec_per_day / warmup
    # 2) if time passed > lr_decay, return min learning rate
    if tp / sec_per_day > lr_decay:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (tp / sec_per_day - warmup) / (lr_decay - warmup)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

# Batch size (i.e. accumulation steps) scheduler
def get_acc_timed(tp):
    if tp / sec_per_day < acc_warmup:
        return min_acc
    if tp / sec_per_day > acc_increase:
        return max_acc
    ratio = (tp / sec_per_day - acc_warmup) / (acc_increase - acc_warmup)
    assert 0 <= ratio <= 1
    return int(np.ceil(min_acc + ratio * (max_acc - min_acc)))

# training loop
X, Y = get_batch('train') # fetch the very first batch
t0 = time.time()
t_init = t0
mfu = -100
time_passed = 0
current_eval_num = 0
local_iter_num = 0 # number of iterations in the lifetime of this process
raw_model = model # unwrap DDP container if needed
running_mfu = -1.0
while True:
    
    # determine and set the learning rate for this iteration
    lr = get_lr_timed(time_passed) if decay_lr else learning_rate
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    gradient_accumulation_steps = get_acc_timed(time_passed) if use_acc_scheduler else gradient_accumulation_steps

    bef_eval = time.time()
    time_passed = bef_eval - t_init
    # evaluate the loss on train/val sets and write checkpoints

    if (time_passed > eval_intervals[current_eval_num] or time_passed > sec_per_day):
        current_eval_num += 1
        losses = estimate_loss()
        train_info[:, current_eval_num - 1] = torch.tensor([iter_num, time_passed, lr, gradient_accumulation_steps, mfu, losses['val']])

        if losses['val'] < best_val_loss or always_save_checkpoint:
            best_val_loss = losses['val']
            if iter_num > 0:
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_args': model_args,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, f'ckpt_{exp_name}.pt'))
        torch.save(train_info, f"{exp_name}.pt")

    # breaking condition
    if time_passed > sec_per_day:
        break

    # offsetting t_init such that the eval time does not count towards the 1 day limit
    t_eval = time.time() - bef_eval
    t_init += t_eval

    

    # forward backward update, with optional gradient accumulation to simulate larger batch size
    # and using the GradScaler if data type is float16
    for micro_step in range(gradient_accumulation_steps):
        with ctx:
            logits, loss = model(X, Y)
            loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
        # immediately async prefetch next batch while model is doing the forward pass on the GPU
        X, Y = get_batch('train')
        # backward pass, with gradient scaling if training in fp16
        scaler.scale(loss).backward()
    # clip the gradient
    if grad_clip != 0.0:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    # step the optimizer and scaler if training in fp16
    scaler.step(optimizer)
    scaler.update()
    # flush the gradients as soon as we can, no need for this memory anymore
    optimizer.zero_grad(set_to_none=True)

    # timing and logging
    t1 = time.time()
    dt = t1 - t0
    t0 = t1
    if iter_num % log_interval == 0:
        #print(torch.cuda.memory_summary())
        # get loss as float. note: this is a CPU-GPU sync point
        # scale up to undo the division above, approximating the true total loss (exact would have been a sum)
        lossf = loss.item() * gradient_accumulation_steps
        if local_iter_num >= 5: # let the training loop settle a bit
            mfu = raw_model.estimate_mfu(batch_size * gradient_accumulation_steps, dt)
            running_mfu = mfu if running_mfu == -1.0 else 0.9*running_mfu + 0.1*mfu
        print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms, mfu {running_mfu*100:.2f}%")
        
    iter_num += 1
    local_iter_num += 1
