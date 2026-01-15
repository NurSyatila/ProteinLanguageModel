# python

import pandas as pd
import os
import argparse
from datasets import Dataset, load_dataset, get_dataset_split_names, DatasetDict

# wget http://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch/fluorescence.tar.gz
# tar -xzvf fluorescence.tar.gz
# Run: python scripts/create_dataset.py -d fluorescence -o fluorescence.csv -t 0.1
# rm -rf fluorescence
# rm fluorescence.tar.gz

def filter_dset(df):
    df['log_fluorescence'] = df['log_fluorescence'].apply(lambda x: x[0] if isinstance(x, list) else x)

    # since the fluorescence data contain protein of various length, we will only use ones with specified sequence length
    # to prevent NaN or missing values on the feature vectors (no issues with embeddings)
    df = df[df['protein_length'] == 237]
    
    return df

def truncate_train(dataset, seed, truncate=1.0):
    if truncate < 1.0:
        truncate_set = dataset.train_test_split(test_size=truncate, seed=seed)
        return truncate_set['test']
    elif truncate == 1.0:
        return dataset
    else:
        print("truncate size must be =< 1.0")
        return dataset    

def main(args):
    seed=42
    dfs = []
    for x in ['train', 'valid', 'test']:
        df = pd.read_json(f'{args.inp_dir}/fluorescence_{x}.json')
        
        df = filter_dset(df)
        df['split'] = x
        #if x == 'train':
        tr_data = Dataset.from_pandas(df)
        df = truncate_train(tr_data, args.seed, truncate=args.truncate)
        df = pd.DataFrame(df)
        dfs.append(df)


    # 2) combine data splits into a file
    result = pd.concat(dfs, ignore_index=True)
    #result.columns = ['primary','protein_length','log_fluorescence','num_mutations','id','split']
    result = result[['primary', 'log_fluorescence', 'split']]
    result.columns = ['sequence','label','split']
    
    #print(len(result))
    #print(len(result[result['split']=='train']))
    
    result.to_csv(args.out_file, index=False)


msg = "Create and truncate fluorescence dataset."
parser = argparse.ArgumentParser(description=msg)
parser.add_argument("-d", "--inp_dir", help = "Path to fluorescence directory containing json files", type=str, required=True)
parser.add_argument("-o", "--out_file", help = "Output csv file", type=str, required=True)
parser.add_argument("-t", "--truncate", help = "Truncate size (0-1.0)", type=float, default=1.0)
parser.add_argument("-s", "--seed", help = "Random seed to use", type=int, default=42)
# Read arguments from command line
args = parser.parse_args()
main(args)



