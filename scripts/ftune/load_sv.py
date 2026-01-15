# python

import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, load_dataset, get_dataset_split_names, DatasetDict, concatenate_datasets

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader

import transformers, datasets
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.t5.modeling_t5 import T5Config, T5PreTrainedModel, T5Stack
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers import T5EncoderModel, T5Tokenizer
from transformers import EsmModel, AutoTokenizer, AutoModelForSequenceClassification

import peft
from peft import get_peft_config, PeftModel, PeftConfig, inject_adapter_in_model, LoraConfig

import numpy as np
import pandas as pd
import os, sys, re, copy
from functools import partial

# This sub-script is used for loading data splits, model, and tokenizer for supervised fine-tuning.
# Supervised fine-tuning can be used for regression (--num_labels 1) and classification (--num_labels 2/n) tasks.
# Use `finetune.py` to set up the overall workflow.
# The input must be whether a CSV file, a directory containing the pre-computed data splits or the path to the HuggingFace Hub dataset.
# This script has been tested with ESM2, Ankh, ProtTrans (ProtT5), and ProstT5 models.
# Fine-tuning for a supervised task requires a sequence classification head on top of the pre-trained language model (PLM).
# ESM2 is already equipped with a classifier head and can be loaded using `AutoModelForSequenceClassification`.
# For T5 models, a custom classifier head will be added on top of the PLM.
# LoRA adapters can be added to the model by setting the `--full False` parameter. 
# The layers are added based on the provided LoRA parameters: `r`, `alpha`, and `target`, which control the number of parameters loaded into the adapters.

# Split the data into train and test
def split_data(dataset, test_size, seed):
    # 80% train + valid, 20% test
    train_test = dataset['all'].train_test_split(test_size=test_size, seed=seed)
    # Split the 80% train + valid in half
    train_valid = train_test['train'].train_test_split(test_size=test_size, seed=seed)
    # gather everyone if you want to have a single DatasetDict
    dataset_splits = DatasetDict({
        'train': train_valid['train'],
        'valid': train_valid['test'],
        'test': train_test['test']})
    return dataset_splits

def truncate_train(dataset, seed, truncate=1.0):
    if truncate < 1.0:
        truncate_set = dataset['train'].train_test_split(test_size=truncate, seed=seed)
        dataset_splits = DatasetDict({
            'train': truncate_set['test'],
            'valid': dataset['valid'],
            'test': dataset['test']})
        return dataset_splits
    elif truncate == 1.0:
        return dataset
    else:
        print("truncate size must be =< 1.0")
        return dataset

def process_sequence(example, checkpoint, seqcol):    
    # Add spaces between each amino acid for ProtT5 and ProstT5 to correctly use them
    if "Rostlab" in checkpoint:
        #example[seqcol] = " ".join(list(re.sub(r"[UZOB]", "X", example[seqcol])))
        example[seqcol] = [" ".join(list(re.sub(r"[UZOB]", "X", seq))) for seq in example[seqcol]]

    # Add <AA2fold> for ProstT5 to inform the model of the input type (amino acid sequence here)
    if "ProstT5" in checkpoint:    
        #example[seqcol] = "<AA2fold>" + " " + example[seqcol]
        example[seqcol] = ["<AA2fold>" + " " + seq for seq in example[seqcol]]
    return example

def tokenize_function(example, tokenizer, seqcol):
    return tokenizer(example[seqcol], max_length=1024, padding=True, truncation=True)


def load_data(inp_path, seed, seqcol, ycol, checkpoint, tokenizer, truncate=1.0):
    df = pd.read_csv(inp_path)
    # convert df into huggingface dataset format
    if 'split' in df.columns:
        train_df = df[df['split'] == 'train']
        val_df = df[df['split'] == 'valid']
        test_df = df[df['split'] == 'test']

        train_dataset = Dataset.from_pandas(train_df.drop(columns=['split']))
        val_dataset = Dataset.from_pandas(val_df.drop(columns=['split']))
        test_dataset = Dataset.from_pandas(test_df.drop(columns=['split']))
        dataset_splits = DatasetDict({
            "train": train_dataset,
            "valid": val_dataset,
            "test": test_dataset
        })

    else:
        train_dataset = Dataset.from_pandas(df)
        dataset = DatasetDict({"all": train_dataset})
        dataset_splits = split_data(dataset, 0.2, seed)

    # Truncate if tr_size < 1.0
    training_dataset = truncate_train(dataset_splits, seed, truncate)

    # Preprocess inputs
    tokenized_dataset = {}
    for split_name, data in training_dataset.items():
        # Preprocess inputs
        processed_ds = training_dataset[split_name].map(partial(process_sequence, checkpoint=checkpoint, seqcol=seqcol), batched=True)
        # Create tokenized dataset 
        tokenized_ds = processed_ds.map(partial(tokenize_function, tokenizer=tokenizer, seqcol=seqcol), batched=True)
        tokenized_dataset[split_name] = tokenized_ds
    
    return tokenized_dataset

def get_model_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return str(params)

class ClassConfig:
    def __init__(self, dropout=0.2, num_labels=1):
        self.dropout_rate = dropout
        self.num_labels = num_labels

class T5EncoderClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, class_config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(class_config.dropout_rate)
        self.out_proj = nn.Linear(config.hidden_size, class_config.num_labels)

    def forward(self, hidden_states):

        hidden_states =  torch.mean(hidden_states,dim=1)  # avg embedding

        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states

class T5EncoderForSimpleSequenceClassification(T5PreTrainedModel):

    def __init__(self, config: T5Config, class_config):
        super().__init__(config)
        self.num_labels = class_config.num_labels
        self.config = config

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        self.dropout = nn.Dropout(class_config.dropout_rate)
        self.classifier = T5EncoderClassificationHead(config, class_config)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.classifier = self.classifier.to(self.encoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.encoder = self.encoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.classifier(hidden_states)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

def load_T5_model(checkpoint, lora_r, lora_alpha, lora_weights, lora_dropout, num_labels, half_precision, full=False):

    # Load model and tokenizer
    if "ankh" in checkpoint :
        model = T5EncoderModel.from_pretrained(checkpoint)

    elif "prot_t5" in checkpoint:
        # possible to load the half precision model (thanks to @pawel-rezo for pointing that out)
        model = T5EncoderModel.from_pretrained(checkpoint)

    elif "ProstT5" in checkpoint:
        model = T5EncoderModel.from_pretrained(checkpoint)

    # Create new Classifier model with PT5 dimensions
    class_config=ClassConfig(num_labels=num_labels)
    class_model=T5EncoderForSimpleSequenceClassification(model.config,class_config)

    # Set encoder and embedding weights to checkpoint weights
    class_model.shared=model.shared
    class_model.encoder=model.encoder

    # Delete the checkpoint model
    model = class_model
    del class_model
    print('full model : ',get_model_params(model))

    if full == True:
        return model

    # lora modification
    #peft_config = LoraConfig(
    #    r=4, lora_alpha=1, bias="all", target_modules=["q","k","v","o"]
    #)
    wdict={"q":"q", "k":"k", "v":"v", "d":"o", "o":"o"}
    target_modules=[wdict[m] for m in list(lora_weights)]
    #target_modules=list(lora_weights)
    peft_config = LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, bias="all", target_modules=target_modules, lora_dropout=lora_dropout
    )
    model = inject_adapter_in_model(peft_config, model) # modifies / manually inserting the model by injecting lora adapter
    #model = get_peft_model(model, peft_config) # automatic handling (might be not be useful for t5-based model)

    # Unfreeze the prediction head
    for (param_name, param) in model.classifier.named_parameters():
                param.requires_grad = True
    print('lora model : ',get_model_params(model))

    return model

#load ESM2 models
def load_esm_model(checkpoint, lora_r, lora_alpha, lora_weights, lora_dropout, num_labels, half_precision, full = False):
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=num_labels, use_safetensors=True)

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
    for (param_name, param) in model.classifier.named_parameters():
                param.requires_grad = True

    print('lora model : ',get_model_params(model))
       
    return model

def load_model(checkpoint, lora_r, lora_alpha, lora_weights, lora_dropout, num_labels=1, mixed = True, full = False):
    # load model
    if "esm" in checkpoint:
        model = load_esm_model(checkpoint, lora_r, lora_alpha, lora_weights, lora_dropout, num_labels, mixed, full)
    else:
        model = load_T5_model(checkpoint, lora_r, lora_alpha, lora_weights, lora_dropout, num_labels, mixed, full)

    return model

def load_tokenizer(checkpoint):
    if "prot_t5" in checkpoint.lower():
        tokenizer = T5Tokenizer.from_pretrained(checkpoint)
    elif "ProstT5" in checkpoint:
        tokenizer = T5Tokenizer.from_pretrained(checkpoint, do_lower_case=False)
    else:
        tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    return tokenizer
