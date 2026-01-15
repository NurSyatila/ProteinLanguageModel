import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
import pathlib
import pandas as pd
import itertools
from Bio import SeqIO
import string
import tqdm
#import ipywidgets as widgets
#from IPython.display import display 
# reference: https://huggingface.co/blog/AmelieSchreiber/mutation-scoring
# three different scoring sources were given as options: wt, masked, msa

import pathlib
import string

import torch

from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, MSATransformer
import pandas as pd
from Bio import SeqIO
import itertools
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# A function to remove insertions (for MSA-transformer only)
def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    # This is an efficient way to delete lowercase characters and insertion characters from a string
    deletekeys = dict.fromkeys(string.ascii_lowercase)
    deletekeys["."] = None
    deletekeys["*"] = None

    translation = str.maketrans(deletekeys)
    return sequence.translate(translation)

# A function to read msa file (for MSA-transformer only)
def read_msa(filename: str, nseq: int) -> List[Tuple[str, str]]:
    """ Reads the first nseq sequences from an MSA file, automatically removes insertions.
    
    The input file must be in a3m format (although we use the SeqIO fasta parser)
    for remove_insertions to work properly."""

    msa = [
        (record.description, remove_insertions(str(record.seq)))
        for record in itertools.islice(SeqIO.parse(filename, "fasta"), nseq)
    ]
    return msa
    
# A function to load model
def load_model(model_location):
    model, alphabet = pretrained.load_model_and_alphabet(model_location)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
        print("Transferred model to GPU")

    batch_converter = alphabet.get_batch_converter()
    return model, alphabet, batch_converter
    
# A function to include calculated score into the dataframe
def label_row(row, sequence, token_probs, alphabet, offset_idx):
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

    wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

    # add 1 for BOS
    score = token_probs[0, 1 + idx, mt_encoded] - token_probs[0, 1 + idx, wt_encoded]
    return score.item()

# A function to compute pseudo-perplexity score (per row)
def compute_pppl(row, sequence, model, alphabet, batch_tokens, offset_idx):
    wt, idx, mt = row[0], int(row[1:-1]) - offset_idx, row[-1]
    assert sequence[idx] == wt, "The listed wildtype does not match the provided sequence"

    # modify the sequence
    sequence = sequence[:idx] + mt + sequence[(idx + 1) :]

    wt_encoded, mt_encoded = alphabet.get_idx(wt), alphabet.get_idx(mt)

    # compute probabilities at each position
    log_probs = []
    for i in range(1, len(sequence) - 1):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = torch.log_softmax(model(batch_tokens_masked)["logits"], dim=-1) #batch_tokens_masked.cuda()
        log_probs.append(token_probs[0, i, alphabet.get_idx(sequence[i])].item())  # vocab size
    return sum(log_probs)

# A function to compute masked-marginals
def compute_masked_marginals(model, alphabet, batch_tokens):
    all_token_probs = []
    for i in range(batch_tokens.size(1)):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, i] = alphabet.mask_idx
        with torch.no_grad():
            token_probs = torch.log_softmax(
                model(batch_tokens_masked)["logits"], dim=-1 #batch_tokens_masked.cuda()
            )
        all_token_probs.append(token_probs[:, i])  # vocab size
    token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
    return (token_probs)

# A function to compute wild type marginals
def compute_wt_marginals(model, batch_tokens):
    with torch.no_grad():
        token_probs = torch.log_softmax(model(batch_tokens)["logits"], dim=-1) #batch_tokens.cuda()
    return (token_probs)

# A function to compute results for MSA-transformer (requires MSA file as additional input)
def msa_shot(df, mutation_col, model, alphabet, batch_converter, offset_idx, msa_path, msa_samples=20):
    # Check if MSA file exists)
    if msa_path.exists():
        data = [read_msa(msa_path, msa_samples)]
        assert (
            scoring_strategy == "masked-marginals"
        ), "MSA Transformer only supports masked marginal strategy"
    
        # Encode the sequence
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
    else:
        print("No MSA_path supplied for MSA-Transformer")

    all_token_probs = []
    for i in range(batch_tokens.size(2)):
        batch_tokens_masked = batch_tokens.clone()
        batch_tokens_masked[0, 0, i] = alphabet.mask_idx  # mask out first sequence
        with torch.no_grad():
            token_probs = torch.log_softmax(
                model(batch_tokens_masked)["logits"], dim=-1 #batch_tokens_masked.cuda()
            )
        all_token_probs.append(token_probs[:, 0, i])  # vocab size
    token_probs = torch.cat(all_token_probs, dim=0).unsqueeze(0)
    df[model_location] = df.apply(
        lambda row: label_row(
            row[mutation_col], sequence, token_probs, alphabet, offset_idx
        ),
        axis=1,
    )
    return df

# A function to create a list containing single residue mutation
def single_mutations(sequence):
    results = []
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    for position in range(1, len(sequence) + 1):
        for i, mt in enumerate(amino_acids):
            wt, idx, mt = sequence[position-1], position, mt
            mutant = wt+str(idx)+mt
            results.append(mutant)
    df = pd.DataFrame(results, columns = ['mutant'])
    return df

# A function to compile all steps and return a new column with zshot scores
def zero_shot(sequence, df, mutation_col, model_location, scoring_strategy, offset_idx=0, msa_path=None, msa_samples=20):
    
    # Load the deep mutational scan
    #df = pd.read_csv(inp_csv)
    
    # Load the model and tokenizer
    model, alphabet, batch_converter = load_model(model_location)

    # Check if MSA_transformer (requires msa file)
    if isinstance(model, MSATransformer):
        df = msa_shot(df, mutation_col, model, alphabet, batch_converter, offset_idx, msa_path, msa_samples=20)
    
    else:
        # Encode the sequence
        data = [ ("protein1", sequence) ]
        batch_labels, batch_strs, batch_tokens = batch_converter(data)
        
        if scoring_strategy == "wt-marginals":
            probs = compute_wt_marginals(model, batch_tokens)
            df[model_location] = df.apply(
                lambda row: label_row(
                    row[mutation_col], sequence, probs, alphabet, offset_idx
                ), axis=1,
            )
            
        elif scoring_strategy == "masked-marginals":
            probs = compute_masked_marginals(model, alphabet, batch_tokens)
            df[model_location] = df.apply(
                lambda row: label_row(
                    row[mutation_col], sequence, probs, alphabet, offset_idx
                ), axis=1,
            )
        
        elif scoring_strategy == "pseudo-ppl":
            df[model_location] = df.apply(
                lambda row: compute_pppl(
                    row[mutation_col], sequence, model, alphabet, batch_tokens, offset_idx
                ),
                axis=1,
            )  
            
        else:
            print("Scoring strategy was not defined.")
    
    return df

# A function to create a log probability heatmap (valid only for single residue mutation)
def visualize_heatmap(sequence, df, mutation_col, model_location, out_path, offset_idx=1, start_pos=None, end_pos=None):
    # The numbering must be based on sequence length, first residue = 0, last residue = len(sequence)
    # The offset_idx will only be used to get the score from the results
    
    # Default handling for start_pos and end_pos
    # Start pos / end pos
    if start_pos is None: 
        start_pos = 0
    if end_pos is None: 
        sequence_length = len(sequence)
        end_pos = sequence_length  # end_pos is inclusive in this case
        ranges = range(start_pos, end_pos)
    else:
        sequence_length = end_pos - start_pos + 1  # calculate the sequence length for the heatmap
        ranges = range(start_pos, end_pos + 1)
    
    # Initialize heatmap with correct dimensions
    heatmap = np.zeros((20, sequence_length))  # Rows for 20 amino acids, columns for the sequence length
    print(f"sequence_length: {sequence_length}, heatmap shape: {heatmap.shape}")

    # Amino acids list
    amino_acids = list("ACDEFGHIKLMNPQRSTVWY")
    
    # Loop over the positions and amino acids
    for position in ranges:  # Adjust to include end_pos
        for i, amino_acid in enumerate(amino_acids):
            # Mutation format: sequence[position-1] (1-based) + str(position) + amino acid
            mutation = sequence[position] + str(position + offset_idx) + amino_acid
            
            # Fetch the score from the dataframe
            score_values = df.loc[(df[mutation_col] == mutation), model_location].values
            if score_values.size > 0:
                score = score_values[0]
                heatmap[i, position - start_pos] = score  # Use 0-based index for heatmap
            else:
                heatmap[i, position - start_pos] = np.nan  # Assign NaN if mutation is not found

    # Visualize the heatmap
    plt.figure(figsize=(15, 5))
    plt.imshow(heatmap, cmap="viridis", aspect="auto", origin="lower")  # 'origin="lower"' for correct y-axis orientation
    plt.xticks(range(sequence_length), list(sequence[start_pos:end_pos+1]))  # Adjust for 0-based index in sequence slice
    plt.tick_params(axis='x', labelsize=5)
    plt.yticks(range(20), amino_acids)
    plt.xlabel("Position in Protein Sequence", )
    plt.ylabel("Amino Acid Mutations")
    plt.title("Predicted Effects of Mutations on Protein Sequence (LLR)")
    plt.colorbar(label="Log Likelihood Ratio (LLR)")
    #plt.show()
    plt.savefig(out_path+".pdf", bbox_inches="tight")

def main(args):
    # Load sequence
    if (args.fasta_file).exists():
        sequence = str(next(SeqIO.parse(args.fasta_file, "fasta")).seq)

        if args.mode == 'input':
            if (args.inp_csv).exists(): 
                df = pd.read_csv(args.inp_csv)
                results = zero_shot(sequence, df, args.mutation_col, args.model_location, args.scoring_strategy, args.offset_idx, msa_path=args.msa_path, msa_samples=args.msa_samples)
                results.to_csv(args.out_path+".csv", index=False)
            else:
                print("Input CSV file does not exist")

        elif args.mode == 'single':
            # Get the list of single residue mutations for the sequence
            offset_idx = 1
            df = single_mutations(sequence)
            results = zero_shot(sequence, df, args.mutation_col, args.model_location, args.scoring_strategy, offset_idx, msa_path=args.msa_path, msa_samples=args.msa_samples)
            results.to_csv(args.out_path+".csv", index=False)
        
        elif args.mode == 'heatmap':
            # only valid for single residue mutations (considering availability of score for all possible combination of wt and mt residues)
            if (args.inp_csv).exists(): 
                df = pd.read_csv(args.inp_csv)
                visualize_heatmap(sequence, df, args.mutation_col, args.model_location, args.out_path, args.offset_idx, args.start_pos, args.end_pos)
            else:
                print("Input CSV file containing score does not exist")
                
        else:
            print("No mode defined: input or single")
    else:
        print("Input fasta file does not exist")

def my_bool(s): return s != 'False' 
def create_parser():
    parser = argparse.ArgumentParser(description="Generate log likelihood ratio heatmap from an ESM model.")
    parser.add_argument("-cp", "--model-location", help = "PyTorch model file OR name of pretrained model to use", type=str, required=True)
    parser.add_argument("-mode", "--mode", help = "'input' (input DMS file), 'single' (single residue mutations generated for given sequence), or 'heatmap'", type=str, choices=["input", "single", "heatmap"], required=True)
    parser.add_argument("-i", "--inp-csv", help = "CSV file containing the deep mutational scan", type=pathlib.Path, default=None)
    parser.add_argument("-fas", "--fasta-file", help = "Fasta file containing the wild-type protein sequence", type=pathlib.Path, required=True)
    parser.add_argument("-mut", "--mutation-col", help = "Column in input file labeling the mutation, format: wt-position-mt", type=str, default="mutant")
    parser.add_argument("-offset", "--offset-idx", help = "Offset of the mutation positions in `--mutation-col`", type=int, default=0)
    parser.add_argument("-o", "--out_path", help = "Path to output files", type=str, required=True)
    parser.add_argument("-score", "--scoring-strategy", help="Scoring strategy", type=str, default="wt-marginals", choices=["wt-marginals", "pseudo-ppl", "masked-marginals"])
    parser.add_argument("-msa","--msa-path", help="path to MSA in a3m format (required for MSA Transformer)", type=pathlib.Path, default=None)
    parser.add_argument("-nmsa", "--msa-samples", type=int, default=400, help="number of sequences to select from the start of the MSA")
    parser.add_argument("-s", "--start-pos", help="Start residue to include in the heatmap", type=int, default=None)
    parser.add_argument("-e", "--end-pos", help="End residue to include in the heatmap", type=int, default=None)

    return parser

if __name__ == "__main__":
    parser = create_parser()
    args = parser.parse_args()
    main(args)

