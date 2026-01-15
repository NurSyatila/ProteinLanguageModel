# /usr/bin/python
# import dependencies 
from ftune.optimize import optimize, plot
import argparse
import sys
import time
import random
import numpy as np
import pandas as pd
import torch
import os
from transformers import set_seed
import accelerate

# This is a script used to fine-tune a protein language model from HuggingFace
# Several arguments need to be set-up

# Two modes of fine-tuning are available:

# (i) Masked Language Modeling (MLM) task:
# MLM is a type of self-supervised learning task where the model learn from homologous sequences by randomly "masked" the amino acids with special tokens.
# The model is trained to predict the masked amino acid based on the surrounding amino acids (context) in the sequence.
# This task allows the model to become more specialized and sensitive to evolutionary patterns and functional features that are shared within families of proteins
# Accuracy and perplexity is uses as the evaluation metric.
# Perplexity describes the model's understanding of the protein sequence (the lower the better) 

# (ii) Supervised task: 
# Supervised fine-tuning is a type of transfer learning where the model is further trained (or fine-tuned) on a labeled dataset to specialize it for a specific downstream task.
# This involves feeding the labeled protein sequences into the model along with their corresponding labels (e.g., class labels, continuous values, or sequence annotations).
# Two modes of supervised tasks are available: (i) Classification (--num_labels 2>) and (ii) Regression (--num_labels 1)

# Optuna is used for hyperparameter (HP) optimization
# HP optimization can improve the performance by finding the best fit to the data and minimizing the error in its predictions compared to the true labels.
# Hyperparameters like the learning rate, batch size, and number of training epochs are typically tuned to optimize the model's performance on the labeled dataset.
# In this case, the learning rate, batch size and validation batch sizes are optimized.

# If 1 GPU is used, the world size must be defined (this will ensure the reproducibility of the training performance)
#os.environ["WORLD_SIZE"] = "1"

def set_randomseeds(s):
    np.random.seed(s)
    random.seed(s)
    set_seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    accelerate.utils.set_seed(s)

def main():
    def my_bool(s): return s != 'False'
    msg = "Perform supervised LoRA-based finetuning of a protein language model on a regression task."
    parser = argparse.ArgumentParser(description=msg)
    # Adding optional argument (otherwise use default)
    # Arguments for traiing modes
    parser.add_argument('-obj', "--objective", help="Finetuning mode: sv (supervised) or mlm (maskedlm)", type=str, required=True)
    parser.add_argument('-default', action='store_true', help="Train the model using defined hyperparameters")
    parser.add_argument('-optim', action='store_true', help="Optimize the hyperparameters using Optuna")
    # Arguments to set data requirements
    parser.add_argument("-i", "--inp_path", help = "Path to input CSV containing sequence and label", type=str, required=True)
    parser.add_argument("-o", "--out_path", help = "Path to log files", type=str, required=True)
    parser.add_argument("-seqcol", "--seqcol", help = "Column describing the sequence", type=str, default="sequence")
    parser.add_argument("-ycol", "--ycol", help = "Column describing the score/y values", type=str, default="label")
    parser.add_argument("-t", "--truncate", help = "Truncate size, cut dataset (percentage) into smaller subset. 1.0 = no truncation. Only valid for 'sv' (supervised) mode.", type=float, default=1.0)
    # Arguments to set model requirements
    parser.add_argument("-cp", "--checkpoint", help = "Model name to use", type=str, required=True)
    parser.add_argument('-f', "--full", help = "Whether to use full model or not", default=True,type=my_bool)
    # Arguments to set LoRA HPs
    parser.add_argument("-lr", "--lora_r", help = "LoRA's r", type=int, default=4)
    parser.add_argument("-la", "--lora_alpha", help = "LoRA's alpha", type=int, default=1)
    parser.add_argument("-lw", "--lora_weights", help = "LoRA's target modules", type=str, default="qkvo") # for prott5=qkvo, esm=
    parser.add_argument("-ld", "--lora_dropout", help = "LoRA's dropout", type=float, default=0.0) 
    # Arguments to set training HPs
    parser.add_argument("-e", "--epochs", help = "Number of epochs", type=int, default=20)
    parser.add_argument("-l", "--lr", help = "Learning rate", type=float, default=1e-5)
    parser.add_argument("-b", "--batch", help = "Batch size", type=int, default=8)
    parser.add_argument("-a", "--accum", help = "Gradient accumulation steps", type=int, default=1)
    parser.add_argument("-v", "--val_batch", help = "Validation batch size", type=int, default=8)
    parser.add_argument("-s", "--seed", help = "Random seed to use", type=int, default=42)
    parser.add_argument("-n", "--num_labels", help = "Number of labels. For regression task, set number of labels=1", type=int, default=1)
    parser.add_argument("-early", "--earlystop", help = "Whether to apply early stopping or not", default=False, type=my_bool)
    # Arguments to set up the optimization step
    parser.add_argument("-nt1", "--n_trials1", help = "Number of Optuna trials for LoRA optimization", type=int, default=20)
    parser.add_argument("-nt2", "--n_trials2", help = "Number of Optuna trials for HP optimization", type=int, default=10)

    # Read arguments from command line
    args = parser.parse_args()

    orig_stdout = sys.stdout
    f = open(f"{args.out_path}.log", "w")
    sys.stdout = f
    start = time.time()

    print("CUDA available:", torch.cuda.is_available())
    print("Number of GPUs:", torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    
    set_randomseeds(args.seed)
    print(args)
    
    if args.full == False: print("PERFORM LORA FINETUNING...")
    else: print("PERFORM FULL MODEL FINETUNING...")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda': torch.cuda.empty_cache()
    print('device:',device)

    if args.objective == 'sv': 
        from ftune.load_sv import load_data, load_tokenizer
        from ftune.train_sv import train
    if args.objective == 'mlm':
        from ftune.load_mlm import load_data, load_tokenizer
        from ftune.train_mlm import train

    # generate dataset
    tokenizer = load_tokenizer(args.checkpoint)
    dataset = load_data(args.inp_path, args.seed, args.seqcol, args.ycol, args.checkpoint, tokenizer, args.truncate)
    print(dataset)

    def compiled_args(lora_r, lora_alpha, lora_weights, lr):
        TrainArgs=[args.out_path, args.checkpoint, tokenizer, dataset, 
                lora_r, lora_alpha, lora_weights, args.lora_dropout, 
                args.num_labels, args.accum, args.batch, args.val_batch, 
                lr, args.epochs, args.seed, args.earlystop, args.full]
        return TrainArgs

    if args.default:
        TrainArgs = compiled_args(args.lora_r, args.lora_alpha, args.lora_weights, args.lr)
        val, history = train(*TrainArgs)
        plot(history[:-1], f"{args.out_path}.pdf")
        print(f"test score: {val}")
        
    elif args.optim:
        # Step 1: Train using default parameters for comparison
        DefArgs = compiled_args(args.lora_r, args.lora_alpha, args.lora_weights, args.lr)
        default_score, defhistory = train(*DefArgs)
        print(f"default score: {default_score}")

        # For lora-based model, optimize the LoRA params and learning rate
        if args.full == False:
            # Step 2: Optimize LoRA params
            TrainArgs1 = compiled_args(None, None, None, args.lr)
            study1 = optimize(args.objective, 'optim_lora', *TrainArgs1, args.n_trials1)

            # Step 3: Optimize learning rate
            TrainArgs2 = compiled_args(study1.best_trial.params['lora_r'], study1.best_trial.params['lora_alpha'], study1.best_trial.params['lora_weights'], None)
            study2 = optimize(args.objective, 'optim_hp', *TrainArgs2, args.n_trials2)

            # Step 4: Check if second-step optimization score is better than the first step, otherwise use default lr
            if study2.best_trial.value > study1.best_trial.value:
                #batch_size = study2.best_trial.params['batch']
                #val_batch_size = study2.best_trial.params['val_batch']
                learning_rate = study2.best_trial.params['lr']
            else:
                #batch_size = args.batch
                #val_batch_size = args.val_batch
                learning_rate = args.lr

            # Step 5: Re-train model with optimized/best params
            TrainArgs = compiled_args(study1.best_trial.params['lora_r'], study1.best_trial.params['lora_alpha'], study1.best_trial.params['lora_weights'], learning_rate)
            optim_score, history = train(*TrainArgs)
            plot(history[:-1], f"{args.out_path}.pdf")
            for trial in study1.trials:
                print(f'(LoRA) Trial {trial.number} - value: {trial.value}; params: {trial.params}')
            for trial in study2.trials:
                print(f'(HPs)  Trial {trial.number} - value: {trial.value}; params: {trial.params}')
            print(f"Optimization Step 1 - LoRA HPs; Best Optuna trial: {study1.best_trial.value}; Best LoRA params: {study1.best_trial.params}")
            print(f"Optimization Step 2 - Training HPs; Best Optuna trial: {study2.best_trial.value}; Best HPs: {study2.best_trial.value}; Default LR: {args.lr}")
            # Step 6: Compare default and best score from Optuna
            if (optim_score > default_score):
                print(f"test score: {optim_score}")
            else:
                print(f"test score: {default_score}")
            
        # For full model, only optimize the learning rate (edit train.py to include some other params for optimization)
        if args.full == True:
            # Step 2: Optimize learning rate, batch size and val batch size
            TrainArgs1 = compiled_args(args.lora_r, args.lora_alpha, args.lora_weights, None)
            study = optimize(args.objective, 'optim_hp', *TrainArgs1, args.n_trials2)
            
            # Step 3: Re-train model with optimized params (Full Model) (this step ensure reproducible results after optimization) 
            TrainArgs = compiled_args(args.lora_r, args.lora_alpha, args.lora_weights, study.best_trial.params['lr'])
            optim_score, history = train(*TrainArgs)
            plot(history[:-1], f"{args.out_path}.pdf")
            for trial in study.trials:
                print(f'Trial {trial.number} - value: {trial.value}; params: {trial.params}')
            print(f"Optimization - Training HPs; Best Optuna trial: {study.best_trial.value}; Best HPs: {study.best_trial.params}; Default LR: {args.lr}")
            if (optim_score > default_score):
                print(f"test score: {optim_score}")
            else:
                print(f"test score: {default_score}")

    else:
        print("No function specified. Use -train or -optimize to run.")

    end = time.time()
    print("Time taken: ", (end-start), "sec.")
    
    sys.stdout = orig_stdout
    f.close()

    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()

if __name__ == "__main__":
    main()

