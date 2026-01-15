# python
# import dependencies
import random
import torch
import numpy as np

import optuna
from optuna.samplers import TPESampler
from functools import partial

import matplotlib.pyplot as plt
import os

# This sub-script defines the Optuna configurations and handles visualization output.
# Use `finetune.py` to set up the overall workflow.
# Update the code to include additional hyperparameters (HPs) for optimization.
# In LoRA mode (--full False), a two-step optimization process is used: 
# LoRA hyperparameters are optimized first, and the best LoRA hyperparameters are then used to search for the best training hyperparameters.
# In full mode (--full True), only the training hyperparameters are optimized.
# If computational resources allow, consider increasing the number of trials to improve the chances of finding better hyperparameters.

def objective(trial, mode, optimmode, out_path, checkpoint, tokenizer, dataset, 
              lora_r, lora_alpha, lora_weights, lora_dropout, 
              num_labels, accum, batch, val_batch, 
              lr, epochs, seed, earlystop, full):
    # Hyperparameters to tune

    if mode == 'sv': from ftune.train_sv import train
    if mode == 'mlm': from ftune.train_mlm import train

    if optimmode=='optim_lora':
        lora_r = trial.suggest_categorical('lora_r', [1, 2, 4, 8, 64])
        lora_alpha = trial.suggest_categorical('lora_alpha', [1, 16, 32, 64, 100])
        lora_weights = trial.suggest_categorical('lora_weights', ['qkvo','q','k','v','o','qk','qv','qo','kv','ko','vo','qkv','qko','qvo','kvo'])
    if optimmode=='optim_hp':
        lr = trial.suggest_float('lr', 1e-7, 1e-2, log=True)
        #batch = trial.suggest_categorical('batch', [8, 16, 64, 128])
        #val_batch = trial.suggest_categorical('val_batch', [8, 16, 64, 128])
        #batch_size = trial.suggest_int('batch_size', 16, 128, step=16)
        #val_batch_size = trial.suggest_int('val_batch_size', 16, 128, step=16)

    TrainArgs=[out_path, checkpoint, tokenizer, dataset, 
              lora_r, lora_alpha, lora_weights, lora_dropout, 
              num_labels, accum, batch, val_batch, 
              lr, epochs, seed, earlystop, full]

    val, history = train(*TrainArgs) 
    
    return val

def optimize(mode, optimmode, out_path, checkpoint, tokenizer, dataset, 
              lora_r, lora_alpha, lora_weights, lora_dropout, 
              num_labels, accum, batch, val_batch, 
              lr, epochs, seed, earlystop, full, n_trials):
    # Hyperparameters to tune

    if optimmode =='optim_lora':
        objective_with_model = partial(objective, mode=mode, optimmode='optim_lora', 
              out_path=out_path, checkpoint=checkpoint, tokenizer=tokenizer, dataset=dataset, 
              lora_r=None, lora_alpha=None, lora_weights=None, lora_dropout=lora_dropout, 
              num_labels=num_labels, accum=accum, batch=batch, val_batch=val_batch, 
              lr=lr, epochs=epochs, seed=seed, earlystop=earlystop, full=full)
    if optimmode =='optim_hp':
        objective_with_model = partial(objective, mode=mode, optimmode='optim_hp', 
              out_path=out_path, checkpoint=checkpoint, tokenizer=tokenizer, dataset=dataset, 
              lora_r=lora_r, lora_alpha=lora_alpha, lora_weights=lora_weights, lora_dropout=lora_dropout, 
              num_labels=num_labels, accum=accum, batch=batch, val_batch=val_batch, 
              lr=None, epochs=epochs, seed=seed, earlystop=earlystop, full=full)
    study = optuna.create_study(direction='maximize', sampler = TPESampler(seed=seed))
    study.optimize(objective_with_model, n_trials=n_trials)

    return study

def plot(history, pdf_output):
    # Get loss, val_loss, and the computed metric from history
    loss = [x['loss'] for x in history if 'loss' in x]
    val_loss = [x['eval_loss'] for x in history if 'eval_loss' in x]

    # Get spearman (for regression) or accuracy value (for classification)
    if [x['eval_accuracy'] for x in history if 'eval_accuracy' in x] != []:
        metric = [x['eval_accuracy'] for x in history if 'eval_accuracy' in x]
    else:
        metric = [x['eval_spearmanr'] for x in history if 'eval_spearmanr' in x]

    epochs = [x['epoch'] for x in history if 'loss' in x]

    fig, host = plt.subplots(figsize=(8,5), layout='constrained')
    ax2 = host.twinx()
    ax3 = host.twinx()
    host.set_xlabel("Epoch")
    host.set_ylabel("Metric")
    ax2.set_ylabel("Train loss")
    ax2.set_xlabel('Epoch')
    ax3.set_ylabel("Validation loss")
    ax3.set_xlabel('Epoch')
    p1 = host.plot(epochs, metric,    color='orange', label="Metric")
    p2 = ax2.plot( epochs, loss,    color='red', label="Train loss")
    p3 = ax3.plot( epochs, val_loss, color='blue', label="Validation loss")
    ax3.spines['right'].set_position(('outward', 60))
    host.yaxis.label.set_color(p1[0].get_color())
    ax2.yaxis.label.set_color(p2[0].get_color())
    ax3.yaxis.label.set_color(p3[0].get_color())
    plt.title("Training History")
    plt.savefig(pdf_output)

