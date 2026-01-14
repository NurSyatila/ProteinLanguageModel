# Protein Language Model
Protein language models, such as ESM-2 is a deep learning model trained on large datasets of protein sequences, learning patterns, relationships, and general features in those sequences. Like natural language processing (NLP) models such as GPT, these models are trained to predict the next amino acid in a sequence, or to understand the context of protein sequences in a way similar to how language models understand words in a sentence.

Protein language models (PLMs) can be extremely powerful tools for binding affinity improvement, particularly because they capture rich, high-dimensional representations of protein sequences. These representations can be leveraged in a variety of ways to predict and improve protein-ligand binding affinity or other properties relevant to molecular interactions. There are several approaches that can be used to achieve this, spanning embeddings, supervised learning, unsupervised learning, and even hybrid strategies. Here are some approaches that will be highlighted in this repository:

1. Supervised Learning on Amino Acid Descriptors (SL-AAFeat): Amino acid descriptors are hand-crafted numerical features that encode known physicochemical, biochemical, or structural properties of amino acids. Each amino acid is mapped to a fixed-length vector (typically 3–10 dimensions), depending on the type of the descriptors. Supervised models trained on handcrafted amino-acid descriptors can serve as a baseline to assess whether improvements arise from learned sequence representations rather than known physicochemical properties.
   
2. Supervised Learning on PLM Embeddings (SL-Embed): Protein language models can generate protein embeddings (dense vector representations) for each protein sequence from the last hidden state of the models, which can be used as input features for downstream tasks like binding affinity prediction. Once a protein sequence is passed through a language model like ESM-2, the output is typically a high-dimensional embedding that encodes various structural, functional, and evolutionary features of the protein. These embeddings can then be used as input features for regression models (e.g., neural networks, support vector machines, random forests) to predict binding affinity.

3. Supervised fine-tuning (FT-SV-Full & FT-SV-LoRA): Supervised learning involves training a model on labeled data, where the goal is to predict a continuous output (e.g., binding affinity) based on input features (e.g., protein sequences and ligand structures). Initially, the model is pre-trained on a large dataset of protein sequences to capture general sequence-function relationships. Then, the pre-trained model can be fine-tuned for specific tasks such as binding affinity regression, using labeled examples where the input is the protein-ligand pair and the output is a continuous binding affinity value. Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning (PEFT) technique designed to adapt large pretrained models while updating only a small subset of parameters. Instead of modifying the original model weights, LoRA introduces two trainable low-rank matrices into selected linear layers. Full-model fine-tuning updates all layers, allowing maximal flexibility and task-specific adaptation. However, this comes at the cost of increased memory usage, longer training times, and a higher risk of catastrophic forgetting—especially when fine-tuning on small or noisy datasets. In contrast, LoRA restricts learning to additive low-rank updates, significantly reducing the number of trainable parameters while preserving the expressive capacity of the pretrained model.

4. Self-supervised fine-tuning (FT-MLM): Fine-tuning a Protein Language Model (PLM) on a masked language model (MLM) task is considered self-supervised, because it doesn’t require manually labeled data. Instead, the model learns from the input sequence by masking parts of the sequence (amino acids, in the case of protein sequences) and learning to predict the masked parts. The model is essentially learning the underlying structure and relationships in the data without needing external supervision (labels) during the learning process.

5. Zero-shot prediction (ZSHOT): Zero-shot prediction using protein language models involves generalizing learned knowledge to new tasks without task-specific fine-tuning, which allow the model to make predictions or perform tasks without having been specifically trained on specific tasks. It leverages the general knowledge the model has acquired during its pre-training phase, where it learns from a vast corpus of protein sequences and their relationships (sequence-function, sequence-structure, etc.) without explicit task-specific labels. 
 
This repository aims to serve a resource for machine learning using available descriptors and protein language models.

The workflow include some machine learning and deep learning approaches for protein function improvement, especially for GFP intensity improvement.

Dataset: GFP protein mutants

Jupyter notebook is provided for each section: 
- ML-Beginners.ipynb: A Step-by-step Guide to Perform Machine Learning on GFP dataset using scikit-learn and GridSearch
- SL-AAFeat.ipynb: A Workflow for Supervised Learning Using Amino Acid Descriptors
- SL-Embed.ipynb : A Workflow for Supervised Learning Using Protein Language Model Embeddings 
- ANN-SV.ipynb: A Workflow for Embedding-based Fine-tuning with Simple Artificial Neural Network
- FT-SV.ipynb : A Workflow for Protein Language Model Fine-tuning with Supervised Learning Objective
- FT-MLM.ipynb : A Workflow for Protein Language Model Fine-tuning with Masked Language Modeling Objective
- ZSHOT.ipynb : A Workflow for Zero-shot Prediction Using ESM-based Models

Here, JupyterLab will be installed and used. JupyterLab is a more advanced and feature-rich interface for Jupyter Notebooks and provides a flexible environment for working with code, data, and visualizations. 

------------------------------------------------------------
1. Create a New Conda Environment (Optional but Recommended)
------------------------------------------------------------
It’s generally a good idea to install JupyterLab in a dedicated environment to avoid conflicts with other packages. To create a new environment, run:
	$ conda env remove -n mlearn -y 
	$ conda create -n mlearn python=3.12
	$ conda activate mlearn

------------------------------------------------------------
2. Install JupyterLab
------------------------------------------------------------

	$ conda install conda-forge::jupyterlab 

------------------------------------------------------------
3. Launch JupyterLab
------------------------------------------------------------
Once JupyterLab is installed, you can start it by running:

	$ jupyter-lab

This will launch JupyterLab in your default web browser. It will typically open at http://localhost:8888 (or another port if 8888 is already in use).

Alternatively, create conda environment from the provided yml file: 

$ conda env create -f mlearn.yml

------------------------------------------------------------
4. Analysis
------------------------------------------------------------
- All codes and data used in this study were made available (scripts/ and data/)
- Installation of required Python packages will be performed on Jupyter Notebook
- To use the scripts, please manually download required packages separately.

------------------------------------------------------------
# ML-Beginners :  A step-by-step guide to perform machine learning on GFP dataset
------------------------------------------------------------
Dataset: data/umetsu/Umetsu_GFP.csv and data/umetsu/Umetsu_GFP_T-scale_pred.csv
Step 1: Data loading and processing
Step 2: Model selection
Step 3: Model construction
Step 4: Model prediction

------------------------------------------------------------
# SL-AAFeat: A workflow to perform machine learning on amino acid features from a descriptor
------------------------------------------------------------
Dataset: data/fluorescence.csv (contains 'sequence' and 'label')

Step 1: Generate Features
$ python scripts/get_aafeat.py -i data/fluorescence.csv -o data/features/ms_whim_scores.csv -f ms_whim_scores -seqcol sequence

------------------------------------------------------------
# SL-AAFeat: A workflow to perform machine learning on amino acid features from a descriptor
------------------------------------------------------------
Step 1: Generate Features

$ python scripts/get_aafeat.py -i data/fluorescence.csv -o data/ms_whim_scores.csv -f ms_whim_scores -seqcol sequence

# Step 2: Supervised Machine Learning
	- Data preprocessing
	- Model selection
	- Hyperparameter optimization and model construction
	- Model prediction

# Run with default parameters
$ python scripts/slearn.py -mode default -i data/ms_whim_scores.csv -o results/ms_whim_scores 

# Run hyperparameter optimization
$ python scripts/slearn.py -mode optim -i data/ms_whim_scores.csv -o results/ms_whim_scores

------------------------------------------------------------
# SL-Embed: A workflow to perform machine learning on embeddings extracted from a protein language model
------------------------------------------------------------
Step 1: Generate embeddings
python scripts/get_embed.py -i data/fluorescence.csv -o data/esm2_t6_8M_UR50D.csv -cp facebook/esm2_t6_8M_UR50D -seqcol sequence

------------------------------------------------------------
# ANN-Embed
------------------------------------------------------------

------------------------------------------------------------
# FT-SV
------------------------------------------------------------

------------------------------------------------------------
# FT-MLM
------------------------------------------------------------

------------------------------------------------------------
# ZSHOT
------------------------------------------------------------
