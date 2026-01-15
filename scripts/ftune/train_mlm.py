# python
from ftune.load_mlm import load_data, load_model

import random
import torch
import numpy as np

import evaluate
from evaluate import load
from datasets import Dataset, load_dataset, DatasetDict

from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import TrainingArguments, Trainer, set_seed, EarlyStoppingCallback, DataCollatorForLanguageModeling
from sklearn.metrics import accuracy_score
import os, math

# This sub-script defines the training setup for MLM task fine-tuning.
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

    #eval_mode = False #obtain initial accuracy/perplexity of the model on the test set before training/finetuning
    print("Model used:", checkpoint, "\n")
    seqcol = None
    ycol = None
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.cuda.empty_cache() 
        mixed = True
        #optimizer = "adamw_bnb_8bit"
    else: 
        mixed = False
        #optimizer = "adamw_torch"

    # set True to use deepspeed
    #deepspeed = False

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
    # Preprocess_logits_for_metrics
    # A function that takes the raw output logits from the model and prepares them into a format suitable for calculating evaluation metrics
    def preprocess_logits_for_metrics(logits, labels):
        if isinstance(logits, tuple):
            # Depending on the model and config, logits may contain extra tensors,
            # like past_key_values, but logits always come first
            logits = logits[0]
        return logits.argmax(dim=-1)

    metric = evaluate.load("accuracy", experiment_id=experiment_id)

    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        # preds have the same shape as the labels, after the argmax(-1) has been calculated
        # by preprocess_logits_for_metrics
        labels = labels.reshape(-1)
        preds = preds.reshape(-1)
        mask = labels != -100
        labels = labels[mask]
        preds = preds[mask]
        return metric.compute(predictions=preds, references=labels)

    # Prepare data collator (randomly mask the tokens)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)

    # Trainer          
    trainer = Trainer(
        model,
        args=args,
        train_dataset=dataset['train'],
        eval_dataset=dataset['valid'],
        processing_class=tokenizer,
        data_collator=data_collator,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        compute_metrics=compute_metrics,
        callbacks = callbacks,
    )    
    
    # Try get initial perplexity
    #init_result = trainer.evaluate(dataset['test'])
    #init_val = init_result['eval_accuracy']
    # Perplexity: A measure of how well a language model predicts a sample
    #try:
    #    init_perplexity = math.exp(init_result["eval_loss"])
    #except OverflowError:
    #    init_perplexity = float("inf")
    #print(f'initial accuracy (perplexity): {init_val} ({init_perplexity})')
    
    # Train model
    trainer.train()
    # Load the test dataset

    # Evaluate
    test_result = trainer.evaluate(dataset['test'])
    val = test_result['eval_accuracy']

    # Perplexity: A measure of how well a language model predicts a sample
    try:
        perplexity = math.exp(test_result["eval_loss"])
    except OverflowError:
        perplexity = float("inf")
    test_result["perplexity"] = perplexity 

    trainer.save_model(logpath)
    trainer.processing_class.save_pretrained(logpath)
 
    print(f"test perplexity: {perplexity}")
    return val, trainer.state.log_history
    
