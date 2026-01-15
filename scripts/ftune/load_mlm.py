# python

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_dataset, DatasetDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader
from Bio import SeqIO

import transformers, datasets
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers import AutoModelForMaskedLM, AutoTokenizer, BitsAndBytesConfig

import peft
from peft import get_peft_config, PeftModel, PeftConfig, inject_adapter_in_model, LoraConfig

import numpy as np
import pandas as pd
import os, sys, re, copy
from functools import partial

# This sub-script is used to load the data splits, model, and tokenizer for MLM task fine-tuning.
# Use `finetune.py` to set up the overall workflow.
# For data splits, use `--split True` to compute random data splits using `train_test_split` from sklearn. 
# In this case, the input must be a CSV file.
# Alternatively, use `--split False` to load pre-computed data splits from a given directory. 
# The input must be a directory containing `_train.fasta`, `_valid.fasta`, and `_test.fasta`.
# This script has been tested only with ESM2 models.
# MLM task fine-tuning is supported for BERT-based models. T5-based models have not been tested for this type of fine-tuning.
# LoRA adapters can be added to the model by setting the `--full False` parameter. 
# The layers are added based on the provided LoRA parameters: `r`, `alpha`, and `target`, which control the number of parameters loaded into the adapters.

def tokenize_function(example, tokenizer, seqcol):
    return tokenizer(example[seqcol], max_length=1024, padding=True, truncation=True)

# Load sequences
def load_fasta_to_dict(fasta_file):
    # Load the sequences from fasta file
    sequences = []
    for record in SeqIO.parse(fasta_file, "fasta-blast"):
        sequences.append({"id": record.id, "sequence": str(record.seq)})

    # Convert the sequences into a Dataset object 
    dataset = Dataset.from_dict({"id": [seq["id"] for seq in sequences],
                "sequence": [seq["sequence"] for seq in sequences]})
    return dataset

# Split the data into train and test
def split_data(dataset, test_size, seed):
    # 80% train + valid, 20% test
    train_test = dataset['train'].train_test_split(test_size=test_size, seed=seed)
    # Split the 80% train + valid in half
    train_valid = train_test['train'].train_test_split(test_size=test_size, seed=seed)
    # gather everyone if you want to have a single DatasetDict
    dataset_splits = DatasetDict({
        'train': train_valid['train'],
        'valid': train_valid['test'],
        'test': train_test['test']})
    return dataset_splits

# Load the data file
def load_data(inp_path, seed, seqcol, ycol, checkpoint, tokenizer, truncate=None):
    # If the input path is a file 
    if os.path.isfile(inp_path):
        print("Given input is a local file. Data splits will be created...")
        dataset = DatasetDict({'train': load_fasta_to_dict(inp_path)})
        dataset_splits = split_data(dataset, 0.2, seed)
    else:
        print("Given input does not exist. Try loading from HuggingFace Hub..")

    # Create tokenized dataset
    tokenized_dataset = dataset_splits.map(partial(tokenize_function, tokenizer=tokenizer, seqcol='sequence'), batched=True)
        
    return tokenized_dataset

def get_model_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return str(params)

#load ESM2 models
def load_model(checkpoint, lora_r, lora_alpha, lora_weights, lora_dropout, num_labels, half_precision, full = False):
    model = AutoModelForMaskedLM.from_pretrained(checkpoint)
    
    print('full model : ',get_model_params(model))
    if full == True:
        return model       
    
    #peft_config = LoraConfig(
    #    r=4, lora_alpha=1, bias="all", target_modules=["query","key","value","dense"]
    #)
    wdict={"q":"query", "k":"key", "v":"value", "d":"dense", "o":"dense"}
    target_modules=[wdict[m] for m in list(lora_weights)]

    peft_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, bias="all", target_modules=target_modules, lora_dropout=lora_dropout
    )
    model = inject_adapter_in_model(peft_config, model)
    
    # Unfreeze the prediction head
    for (param_name, param) in model.lm_head.named_parameters():
                param.requires_grad = True

    print('lora model : ',get_model_params(model))
       
    return model

def load_tokenizer(checkpoint):
    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return tokenizer
