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
- ML-Beginners.ipynb: A Step-by-step Guide to Perform Machine Learning on GFP dataset using scikit-learn and GridSearchCV
- SL-AAFeat.ipynb: A Workflow for Supervised Learning Using Amino Acid Descriptors
- SL-Embed.ipynb : A Workflow for Supervised Learning Using Protein Language Model Embeddings 
- ANN-SV.ipynb: A Workflow for Embedding-based Fine-tuning with Simple Artificial Neural Network
- FT-SV.ipynb : A Workflow for Protein Language Model Fine-tuning with Supervised Learning Objective
- FT-MLM.ipynb : A Workflow for Protein Language Model Fine-tuning with Masked Language Modeling Objective
- ZSHOT.ipynb : A Workflow for Zero-shot Prediction Using ESM-based Models

Here, JupyterLab will be installed and used. JupyterLab is a more advanced and feature-rich interface for Jupyter Notebooks and provides a flexible environment for working with code, data, and visualizations. 


### 1. Create a New Conda Environment (Optional but Recommended)
   
It’s generally a good idea to install JupyterLab in a dedicated environment to avoid conflicts with other packages. To create a new environment, run:
	```
	conda env remove -n mlearn -y
	conda create -n mlearn python=3.12
	conda activate mlearn
	```

### 2. Install JupyterLab

	```conda install conda-forge::jupyterlab``` 

### 3. Launch JupyterLab
Once JupyterLab is installed, you can start it by running:

	```jupyter-lab```

This will launch JupyterLab in your default web browser. It will typically open at http://localhost:8888 (or another port if 8888 is already in use).

Alternatively, create conda environment from the provided yml file: 

	```conda env create -f mlearn.yml```

### 4. Analysis
	- All codes and data used in this study were made available (scripts/ and data/)
	- Installation of required Python packages will be performed on Jupyter Notebook
	- To use the scripts, please manually download required packages separately.

## ML-Beginners :  A step-by-step guide to perform machine learning on GFP dataset

Dataset: data/umetsu/Umetsu_GFP.csv and data/umetsu/Umetsu_GFP_T-scale_pred.csv
	Step 1: Data loading and processing
	Step 2: Model selection (ms)
	Step 3: Model construction (mc)
	Step 4: Model prediction (mp)

**Run the workflow**
	Available modes: ms, mc, mp or workflow to run all steps
	Available features: blosum_indices fasgai_vectors ms_whim_scores protfp_descriptors st_scales t_scales vhse_scale z_scales 
						(and more, refer to https://github.com/althonos/peptides.py)

	```
	python scripts/GFP_practice.py -mode workflow \
									  -i data/umetsu/Umetsu_GFP.csv \
									  -p data/umetsu/Umetsu_GFP_T-scale_pred.csv \
									  -o results/GFP_practice_gpr \
									  -feature t_scales \
									  -model gpr
	```
	
## SL-AAFeat: A workflow to perform machine learning on amino acid features from a descriptor

Dataset: data/fluorescence.csv (contains 'sequence' and 'label')

**Step 1: Generate Features**  (default sequence column: 'sequence')

```
python scripts/get_aafeat.py -i data/fluorescence.csv \
						        -o data/features/ms_whim_scores.csv \
                                -f ms_whim_scores -seqcol sequence
```

**Step 2: Supervised Machine Learning**
	- Data preprocessing
	- Model selection
	- Hyperparameter optimization and model construction
	- Model evaluation

*Run with default parameters (default random seed = 42)*
```python scripts/slearn.py -mode default -i data/ms_whim_scores.csv -o results/ms_whim_scores_default```

*Run hyperparameter optimization (default number of trials, nt = 30, default random seed = 42)*
```python scripts/slearn.py -mode optim -i data/ms_whim_scores.csv -o results/ms_whim_scores_optim -nt 5```

## SL-Embed: A workflow to perform machine learning on embeddings extracted from a protein language model

**Step 1: Generate embeddings** (default sequence column: 'sequence')
Three different pooling strategies are available: mean (default), max, sum.

```python scripts/get_embed.py -i data/fluorescence.csv -o data/esm2_t6_8M_UR50D_mean.csv -cp facebook/esm2_t6_8M_UR50D -p mean```

**Step 2: Supervised Machine Learning**
	- Data preprocessing
	- Model selection
	- Hyperparameter optimization and model construction
	- Model evaluation

*Run with default parameters (default random seed = 42)*
```python scripts/slearn.py -mode default -i data/esm2_t6_8M_UR50D_mean.csv -o results/esm2_t6_8M_UR50D_mean_default```

*Run hyperparameter optimization (default number of trials, nt = 30, default random seed = 42)*
```python scripts/slearn.py -mode optim -i data/esm2_t6_8M_UR50D_mean.csv -o results/esm2_t6_8M_UR50D_mean_optim -nt 5```

## FT: Protein Language Model Finetuning with Supervised Learning Task (SV) or Masked Language Modeling Task (MLM)

	- Two fine-tuning objectives are available: supervised learning (--objective sv) and masked language modeling (--objective mlm)
	- Two training modes are available: using default hyperparameters (-default) and perform hyperparameter optimization (-optim)
	- Two types are available: full model training (--full True) and LoRA-based model training (--full False)

**Steps:**
	- Model and tokenizer loading
	- Data processing
	- Model training and hyperparameter optimization
	- Model evaluation

*(A) Run for both LoRA and full-based models on supervised learning tasks:*

*#Example 1: A single run*
```
python scripts/finetune.py -obj sv -default \
						   -f True -i data/fluorescence.csv \
                           -o results/esm2_t6_8M_UR50D_sv_full_default \
						   -cp facebook/esm2_t6_8M_UR50D -nt1 3 -nt2 3
```

```
python scripts/finetune.py -obj mlm -default \
                           -f True -i data/fluorescence_homologs.fasta \
 				           -o results/esm2_t6_8M_UR50D_mlm_full_default \
                           -cp facebook/esm2_t6_8M_UR50D -nt1 3 -nt2 3
```

*#Example 2: For loop run (scripts/finetune.sh)*
#Define the function to perform the task (sv or mlm)
```
function finetune {
	objective="$1"
	data="$2"
	checkpoint="facebook/esm2_t6_8M_UR50D"
	embed="$(basename "${checkpoint%.*}")"
	for action in default optim; do
		full=$(if [[ $size == "full" ]]; then echo "True"; else echo "False"; fi)
		for size in full lora; do
			outpath="results/${objective}_${embed}_${size}_${action}"
			python scripts/finetune.py \
                                -mode $objective -${action} -f $full -i $data \
                                -o $outpath -cp $checkpoint -nt1 3 -nt2 3		
		done
	done
}
```
Run the function with arguments in command line / terminal

*Supervised learning task*
```sh scripts/finetune.sh finetune sv data/fluorescence.csv```

*Masked language modeling task*
```sh scripts/finetune.sh finetune mlm data/fluorescence_homologs.fasta```

## ZSHOT

Three different modes are available:
	- input: Compute score based on input CSV file containing the deep mutational scan
	- single: Compute score based on single residue mutations generated for given wild-type sequence
	- heatmap: Generate heatmap of pre-computed scores from a given input CSV file and for a specified range of residue numbers

**(A) mode = 'input'**

- The original data use offset of 24, i.e. the first residue start at 24 and not 0 or 1
- A new column 'esm2_t6_8M_UR50D' will be generated that store the generated scores 

```
python scripts/zero_shot.py -cp esm2_t6_8M_UR50D \
		-mode input \
		-i data/zshot/BLAT_ECOLX_Ranganathan2015.csv \
		-fas data/zshot/BLAT_ECOLX_Ranganathan2015.fasta \
		-offset 24 \
		-o results/zshot_BLAT_ECOLX_Ranganathan2015_esm2_t6_8M_UR50D \
		-score masked-marginals
```

**(B) mode = 'single'**

Scores will be computed for all possible combinations of wild-type and mutant residues

```
python scripts/zero_shot.py -cp esm2_t6_8M_UR50D \
		-mode single \
		-fas data/zshot/BLAT_ECOLX_Ranganathan2015.fasta \
		-o results/zshot_single_esm2_t6_8M_UR50D \
		-score masked-marginals
```

**(C) mode = 'heatmap'**

- First residue start with 0
- The offset_idx is only used to get score from the data

#Try for all residues

```
python scripts/zero_shot.py -cp esm2_t6_8M_UR50D \
		-mode heatmap \
		-i results/zshot_single_esm2_t6_8M_UR50D.csv \
		-fas data/zshot/BLAT_ECOLX_Ranganathan2015.fasta \
		-offset 1 \
		-o results/zshot_single_esm2_t6_8M_UR50D_heatmap \
		-score masked-marginals
```

#Try for residue 0-9

```
python scripts/zero_shot.py -cp esm2_t6_8M_UR50D \
		-mode heatmap \
		-i results/zshot_single_esm2_t6_8M_UR50D.csv \
		-fas data/zshot/BLAT_ECOLX_Ranganathan2015.fasta \
		-offset 1 \
		-o results/zshot_single_esm2_t6_8M_UR50D_heatmap_0_9 \
		-score masked-marginals \
		-s 0 -e 9
```
