# python
from ftune.load_sv import load_model

import random
import torch
import numpy as np

from evaluate import load
from datasets import Dataset, load_dataset, DatasetDict

from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer, set_seed, EarlyStoppingCallback
from transformers import EsmModel, AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import accuracy_score
import os

# This sub-script defines the training setup for supervised fine-tuning.
# Use `finetune.py` to set up the overall workflow.
# DeepSpeed was not used in this case.

# Note: Training arguments
# For training, # we recommend an effective batch size of 8
# Gradient accumulation: # effective training batch size is batch * accum
# Val_batch = batch size for evaluation
# Mixed = True: enable mixed precision training
# Full = True: enable training of the full model (instead of LoRA)

# Main training fuction
def train(
        logpath, checkpoint, tokenizer, dataset,                        # model checkpoint
        lora_r=4, lora_alpha=1, lora_weights='qkvo', lora_dropout=0.0,  # lora arguments
        num_labels = 1, accum = 1, batch = 8, val_batch = 8,            # training arguments    
        lr = 3e-4, epochs = 20, seed = 42,                              # training arguments
        earlystop = False, full = False):                               # training arguments

    print("Model used:", checkpoint, "\n")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.empty_cache() 
        mixed = True
    #    optimizer = "adamw_bnb_8bit"
    else: 
        mixed = False
    #    optimizer = "adamw_torch"
    
    # set True to use deepspeed
    #deepspeed = False

    # Correct incompatible training settings
    if "ankh" in checkpoint and mixed:
        print("Ankh models do not support mixed precision training!")
        print("switched to FULL PRECISION TRAINING instead")
        mixed = False
    
    # Set all random seeds
    #set_seeds(seed)

    # Conditionally create the list of callbacks
    callbacks = []
    if earlystop: 
        print("Early stopping implemented.")
        save_strategy = "epoch"
        load_best_model_at_end=True
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=5))
    else: 
        print("Early stopping NOT implemented.")
        save_strategy = "no"
        load_best_model_at_end=False

    if "esm" in checkpoint: 
        save_safetensors=True
    else: 
        save_safetensors=False    
    #steps_per_epoch = math.ceil(len(train_dataset) / batch) 

    # Huggingface Trainer arguments
    args = TrainingArguments(
        logpath,                                        # Output directory
        eval_strategy = "epoch",                        # Evaluation strategy (evaluate at the end of each epoch)
        logging_strategy = "epoch",                     # Logging strategy (log at the end of each epoch)
        learning_rate=lr,                               # Base learning rate
        per_device_train_batch_size=batch,              # Batch size for training
        per_device_eval_batch_size=val_batch,           # Batch size for evaluation
        gradient_accumulation_steps=accum,
        num_train_epochs=epochs,                        # # Total number of training epochs
        seed = seed,
        fp16 = mixed,
        save_strategy = save_strategy,                  # Save model for every epoch
        metric_for_best_model = 'eval_loss',
        load_best_model_at_end=load_best_model_at_end,
        save_total_limit = 2,
        warmup_steps=500,                               # Number of steps for learning rate warmup
        lr_scheduler_type="cosine",                     # Set the scheduler type to "linear", "cosine", "constant", or "polynomial"
        save_safetensors=save_safetensors
    )

    # Load model
    model = load_model(checkpoint, lora_r, lora_alpha, lora_weights, lora_dropout, num_labels, mixed, full)
    
    # Metric definition for validation data
    experiment_id="_".join(logpath.split("/")[-4:])

    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids

        # Handle tuple outputs
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        # Remove ignored labels
        if labels.ndim > 1:
            labels = labels.squeeze()

        if predictions.ndim > 1 and predictions.shape[-1] == 1:
            predictions = predictions.squeeze()

        mask = labels != -100
        predictions = predictions[mask]
        labels = labels[mask]

        if num_labels > 1:  # classification
            metric = load("accuracy", experiment_id=experiment_id)
            predictions = np.argmax(predictions, axis=1)
        else:  # regression
            metric = load("spearmanr", experiment_id=experiment_id)

        return metric.compute(predictions=predictions, references=labels)


    # Trainer          
    trainer = Trainer(
        model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        processing_class=tokenizer,
        compute_metrics=compute_metrics,
        callbacks = callbacks,
    )    
    
    # Train model
    trainer.train()
    # Load the test dataset

    test_results = trainer.evaluate(dataset['test'])
    if num_labels>1:  # for classification
        val = test_results['eval_accuracy']
    else:
        val = test_results['eval_spearmanr']
    return val, trainer.state.log_history

