# python
#import dependencies
import peptides
import pandas as pd
import argparse

# This script is used for feature extraction from an amino acid descriptor

def compute_features(df, seqcol, feature):
    # Available features: available: blosum_indices fasgai_vectors ms_whim_scores protfp_descriptors st_scales t_scales vhse_scale z_scales
    # and many more (refer to peptides package in github: https://github.com/althonos/peptides.py)

    # Get features from a descriptor
    data1=[[list(getattr(peptides.Peptide(a_a), feature)()) for a_a in list(seq)] for seq in df[seqcol]]
    data2=[[s for j in k for s in j] for k in data1]

    # Retrieve features (X) and target values (y)
    X = pd.DataFrame(data2)
    
    return X

def main(args):
    df = pd.read_csv(args.inp_file)
    # Get features for all sequences in the dataset (X)
    X = compute_features(df, args.seqcol, args.feature)

    # Combine the columns
    cols = [n for n in df.columns if n != args.seqcol]
    features = pd.concat([df[args.seqcol], X, df[cols]],axis=1)

    # Save to csv file
    features.to_csv(args.out_file, index=False)

msg = "Compute amino acid features from a given descriptor. Available descriptors: blosum_indices fasgai_vectors ms_whim_scores protfp_descriptors st_scales t_scales vhse_scale z_scales and many more (refer to peptides package in github: https://github.com/althonos/peptides.py)"
parser = argparse.ArgumentParser(description=msg)
# Adding optional argument (otherwise use default)
parser.add_argument("-i", "--inp_file", help = "Path to input CSV/JSON file containing sequence and label", type=str, required=True)
parser.add_argument("-o", "--out_file", help = "Path to output CSV/JSON to store the features", type=str, required=True)
parser.add_argument("-f", "--feature", help = "Descriptor to compute amino acid features", type=str, required=True)
parser.add_argument("-seqcol", "--seqcol", help = "Column describing the sequence", type=str, default='sequence')


# Read arguments from command line
args = parser.parse_args()
main(args)

# python scripts/get_aafeat.py -i data/fluorescence.csv -o data/ms_whim_scores.csv -f ms_whim_scores -seqcol sequence