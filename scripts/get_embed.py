# python
#import dependencies
import torch
import numpy as np
import pandas as pd
import time
import random
import itertools
import sys
import os
from transformers import AutoTokenizer, AutoModel, EsmModel, T5Tokenizer, T5EncoderModel, BertModel, BertTokenizer, RoFormerModel, RoFormerTokenizer
import argparse

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# setup model list according to model type
pooling_methods = ['mean', 'max', 'sum', 'global']

InstalledModels = [ "proteinbert", "tape", "protflash", "esm3s", "esmc_300m", "esmc_600m"]
TransformerModels = [ 
         "facebook/esm2_t6_8M_UR50D", "facebook/esm2_t12_35M_UR50D","facebook/esm2_t30_150M_UR50D", "facebook/esm2_t33_650M_UR50D" , "facebook/esm2_t36_3B_UR50D",
         "facebook/esm1v_t33_650M_UR90S_1", "facebook/esm1v_t33_650M_UR90S_2",
         "facebook/esm1v_t33_650M_UR90S_3", "facebook/esm1v_t33_650M_UR90S_4",
         "facebook/esm1v_t33_650M_UR90S_5",
         "Rostlab/prot_t5_xl_bfd", "Rostlab/prot_t5_xl_uniref50", "Rostlab/Prostt5",
         "Rostlab/prot_bert", "Rostlab/prot_bert_bfd",
         "ElnaggarLab/ankh-base" , "ElnaggarLab/ankh-large"]

# setup model list according to input data
SpacesInput = [ "Rostlab/prot_t5_xl_bfd", "Rostlab/prot_t5_xl_uniref50", 
                "Rostlab/prot_bert", "Rostlab/prot_bert_bfd" ] 
DFInput = ["facebook/esm2_t6_8M_UR50D", "facebook/esm2_t12_35M_UR50D","facebook/esm2_t30_150M_UR50D", "facebook/esm2_t33_650M_UR50D" ,
         "facebook/esm2_t36_3B_UR50D",
         "facebook/esm1v_t33_650M_UR90S_1",
         "facebook/esm1v_t33_650M_UR90S_2",
         "facebook/esm1v_t33_650M_UR90S_3",
         "facebook/esm1v_t33_650M_UR90S_4",
         "facebook/esm1v_t33_650M_UR90S_5",
         "ElnaggarLab/ankh-base" , "ElnaggarLab/ankh-large", "proteinbert", "tape", "esm3s", "esmc_300m", "esmc_600m", "saprot"]
SpecialInput = ["Rostlab/Prostt5", "protflash" ]

# 1) preprocess input data
def seq_preprocess(df, seqcol, checkpoint):
    if checkpoint in SpacesInput: 
        df[seqcol]=df.apply(lambda row : " ".join(row[seqcol]), axis = 1)
        return df    
        
    elif checkpoint in DFInput: 
        return df 
        
    elif checkpoint in SpecialInput: 
        if checkpoint == "Rostlab/Prostt5":
            df[seqcol]=df.apply(lambda row : "<AA2fold>" + " " + " ".join(row[seqcol]), axis = 1)
            #df[seqcol]=df.apply(lambda row : "<AA2fold> " + row["sequence"], axis = 1) 
            return df
        
        if checkpoint == "protflash":
            data = df[[seqcol ,seqcol]].values.tolist()
            data = [tuple(s) for s in data]
            return data 
    
    else: 
        return None

def compute_embeddings(df, seqcol, checkpoint, pool):
    data = seq_preprocess(df, seqcol, checkpoint)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # a) set up transformer-based models and get embeddings
    if checkpoint in TransformerModels:
        # i) load model and tokenizer
        if "ankh" in checkpoint:
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            model = T5EncoderModel.from_pretrained(checkpoint).to(device)
            if device.type == 'cuda':
                model = model.to(device)
        
        elif "t5" in checkpoint.lower():
            tokenizer = T5Tokenizer.from_pretrained(checkpoint)
            model = T5EncoderModel.from_pretrained(checkpoint).to(device)
            if device.type == 'cuda':
                model = model.half()

        else: #esm/bert/rotamer
            tokenizer = AutoTokenizer.from_pretrained(checkpoint)
            model = AutoModel.from_pretrained(checkpoint).to(device)
            if device.type == 'cuda': 
                model = model.half()
        # ii) get embeddings
        emb = []
        for i in range(0,len(data)):
            inputs = tokenizer(data[seqcol].loc[i], return_tensors="pt", max_length = 2400, truncation=True, padding=False).to(device)
            with torch.no_grad():
                # compute single seq embedding, calculate mean across seq len dimension, transform to np array
                #emb.append( np.array( torch.mean( model(**inputs).last_hidden_state.cpu(), dim = 1)))
                outputs = model(**inputs).last_hidden_state.cpu()
                if pool == 'mean':
                    embedding = torch.mean(outputs, dim=1)
                elif pool == 'sum':
                    embedding = torch.sum(outputs, dim=1)
                elif pool == 'max':
                    embedding, _ = torch.max(outputs, dim=1)
                else:
                    raise ValueError(f"Unsupported pooling method: {pool} (mean, max, sum)")
            emb.append(np.asarray(embedding))
        embeddings = pd.DataFrame(np.concatenate(emb))

    else:
        if checkpoint == "proteinbert": #kiv first
            # pooling method not mentioned
            import tensorflow as tf
            # Hide GPU from visible devices (use CPU)
            tf.config.set_visible_devices([], 'GPU')
            from proteinbert import load_pretrained_model
            from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
            batch_size = 32
            seq_len = data[seqcol].str.len().max() + 2
            sequences = data[seqcol].values.tolist()
            pretrained_model_generator, input_encoder = load_pretrained_model()
            model = get_model_with_hidden_layers_as_outputs(pretrained_model_generator.create_model(seq_len))
            encoded_x = input_encoder.encode_X(sequences, seq_len)
            local_representations, global_representations = model.predict(encoded_x, batch_size=batch_size)
            # pooling performed on local_representations
            
            outputs = torch.tensor(local_representations)
            if pool == 'mean':
                embedding = torch.mean(outputs, dim=1)
            elif pool == 'sum':
                embedding = torch.sum(outputs, dim=1)
            elif pool == 'max':
                embedding, _ = torch.max(outputs, dim=1)
            elif pool == 'global':
                embedding= global_representations
            else:
                raise ValueError(f"Unsupported pooling method: {pool} (mean, max, sum, global)")

            embeddings = pd.DataFrame(embedding)
        
        if checkpoint == "tape":
            # manual average pooling over the residue (token) embeddings and and it specifically includes the special tokens (i.e., [CLS] and [SEP] from BERT)
            from tape import ProteinBertModel, TAPETokenizer
            model = ProteinBertModel.from_pretrained('bert-base')
            tokenizer = TAPETokenizer(vocab='iupac')
            emb = []
            for i in range(0,len(data)):
                token_ids = torch.tensor([tokenizer.encode(data[seqcol].loc[i])])
                output = model(token_ids)
                sequence_output = output[0][0]
                if pool == 'mean':
                    embedding = torch.mean(sequence_output, dim=0)
                elif pool == 'sum':
                    embedding = torch.sum(sequence_output, dim=0)
                elif pool == 'max':
                    embedding, _ = torch.max(sequence_output, dim=0)
                else:
                    raise ValueError(f"Unsupported pooling method: {pool} (mean, max, sum)")

                emb.append( embedding.detach().numpy() )
            embeddings = pd.DataFrame(emb)

        if checkpoint == "protflash":
            from ProtFlash.pretrain import load_prot_flash_base
            from ProtFlash.utils import batchConverter
            
            # revise script for batches of data
            model = load_prot_flash_base()
            BATCH_SIZE = 60
            emb = []
            # batch processing
            for j in range(0, len(data), BATCH_SIZE):
                batch_data = data[j:j+BATCH_SIZE]
                ids, batch_token, lengths = batchConverter(batch_data)
                with torch.no_grad():
                    token_embedding = model(batch_token, lengths)
                # Generate per-sequence representations via averaging
                for i, (_, seq) in enumerate(batch_data):
                    seq_emb = token_embedding[i, 0:len(seq) + 1]
                    if pool == 'mean':
                        embedding = seq_emb.mean(0)
                    elif pool == 'sum':
                        embedding = seq_emb.sum(0)
                    elif pool == 'max':
                        embedding = seq_emb.max(0).values
                    else:
                        raise ValueError(f"Unsupported pooling method: {pool} (mean, max, sum)")
                    emb.append(embedding)
            emb = torch.stack(emb, dim=0).numpy()
            embeddings = pd.DataFrame(emb)

        if checkpoint == "esm3s":
            # options: esm3-small-2024-08 (1.4b), esm3-medium-2024-08 (7b), esm3-large-2024-03 (98b)
            from esm.models.esm3 import ESM3
            from esm.sdk.api import ESMProtein, SamplingConfig
            from esm.utils.constants.models import ESM3_OPEN_SMALL
            client = ESM3.from_pretrained(ESM3_OPEN_SMALL).to("cuda")
            emb = []
            for i in range(0,len(data)):
                protein = ESMProtein(sequence=data[seqcol].loc[i])
                protein_tensor = client.encode(protein)
                output = client.forward_and_sample(
                    protein_tensor, SamplingConfig(return_per_residue_embeddings=True)
                )
                #embedding = np.mean(np.array(output.per_residue_embedding), axis=0)
                sequence_output = torch.tensor(output.per_residue_embedding)
                if pool == 'mean':
                    embedding = torch.mean(sequence_output, dim=0)
                elif pool == 'sum':
                    embedding = torch.sum(sequence_output, dim=0)
                elif pool == 'max':
                    embedding, _ = torch.max(sequence_output, dim=0)
                else:
                    raise ValueError(f"Unsupported pooling method: {pool} (mean, max, sum)")

                emb.append( embedding.detach().cpu().numpy() )
            embeddings = pd.DataFrame(np.asarray(emb))


        if "esmc" in checkpoint:
            #options: esmc-300m-2024-12, esmc-600m-2024-12, esmc-6b-2024-12
            from esm.models.esmc import ESMC
            from esm.sdk.api import ESMProtein, LogitsConfig
            client = ESMC.from_pretrained(checkpoint).to("cuda")
            emb = []
            for i in range(0,len(data)):
                protein = ESMProtein(sequence=data[seqcol].loc[i])
                protein_tensor = client.encode(protein)
                logits_output = client.logits(
                    protein_tensor, LogitsConfig(sequence=True, return_embeddings=True)
                )
                #residue_embedding = np.squeeze(np.array(logits_output.embeddings), axis=0)
                outputs = torch.tensor(logits_output.embeddings)
                if pool == 'mean':
                    embedding = torch.mean(outputs, dim=1)
                elif pool == 'sum':
                    embedding = torch.sum(outputs, dim=1)
                elif pool == 'max':
                    embedding, _ = torch.max(outputs, dim=1)
                else:
                    raise ValueError(f"Unsupported pooling method: {pool} (mean, max, sum)")
                #emb.append( embedding.detach().cpu().numpy())
                emb.append(np.asarray(embedding.cpu().squeeze()))
            embeddings = pd.DataFrame(emb)

    return embeddings

def main(args):
    df = pd.read_csv(args.inp_file)

    # Get features for all sequences in the dataset (X)
    X = compute_embeddings(df, args.seqcol, args.checkpoint, args.pool)

    # Combine the columns
    cols = [n for n in df.columns if n != args.seqcol]
    features = pd.concat([df[args.seqcol], X, df[cols]],axis=1)

    # Save to csv file
    features.to_csv(args.out_file, index=False)

    print("completed!")

msg = "A script to perform feature-based (embedding) finetuning.\nUsage: get_embed.py -csv data/train1 -log results/train1/embed_log1"
parser = argparse.ArgumentParser(description=msg)
# Adding optional argument (otherwise use default)
parser.add_argument("-i", "--inp_file", help = "Path to input CSV", type=str, required=True)
parser.add_argument("-o", "--out_file", help = "Path to output CSV", type=str, required=True)
parser.add_argument("-cp", "--checkpoint", help = "Model to use", type=str, required=True)
parser.add_argument("-seqcol", "--seqcol", help = "Sequence column", type=str, default="sequence")
parser.add_argument("-p", "--pool", help = "Pooling method. Available: mean, sum, max", type=str, default="mean")

# Read arguments from command line
args = parser.parse_args()
main(args)

#python scripts/get_embed.py -i data/fluorescence.csv -o data/esm2_t6_8M_UR50D.csv -cp facebook/esm2_t6_8M_UR50D -seqcol sequence
