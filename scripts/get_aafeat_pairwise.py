# python
#import dependencies
import numpy as np
import pandas as pd
import random
import itertools
import sys
import re
import glob
import os

# combine two datasets
# e.g. Z-scale and T-scale

def generate_pairwise_file(filepath, aafeat1, aafeat2):
    df1 = pd.read_csv(f"{filepath}/{aafeat1}.csv")
    df2 = pd.read_csv(f"{filepath}/{aafeat2}.csv")
    if 'split' in df1.columns:
        df_feat = pd.concat([df1.iloc[: , :-2], df2.iloc[: , 1:]], axis=1)
    else:
        df_feat = pd.concat([df1.iloc[: , :-1], df2.iloc[: , 1:]], axis=1)
    outfile=f"{filepath}/{aafeat1}_AND_{aafeat2}.csv"
    df_feat.to_csv(outfile, index=False)


#filepath="data/vhh/aafeat"
filepath=sys.argv[1]
desc_list=["blosum_indices", "fasgai_vectors", "ms_whim_scores", "protfp_descriptors", "st_scales", "t_scales", "vhse_scales", "z_scales"]
#desc_list = ["BLOSUM", "FASGAI", "MS-WHIM", "T-scale", "ST-scale", "Z-scale", "VHSE", "ProtFP"]
pairwise_list = [list(n) for n in itertools.combinations(desc_list, 2)]
for aafeat1, aafeat2 in pairwise_list:
    print(aafeat1, aafeat2)
    generate_pairwise_file(filepath, aafeat1, aafeat2)

