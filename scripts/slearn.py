# python
# import dependencies
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
from sklearn.metrics import root_mean_squared_error
import optuna
import os, sys, argparse, time, pickle
import numpy as np
from optuna_integration.sklearn import OptunaSearchCV
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from datasets import Dataset, load_dataset, get_dataset_split_names, DatasetDict, concatenate_datasets

#from sklearnex import patch_sklearn, config_context, set_config
#from sklearnex import patch_sklearn
#patch_sklearn()
#set_config(target_offload="cuda")

# This code is used for supervised learning using SVR on pre-computed features
# Bayesian optimization from Optuna is used for hyperparameter (HP) tuning that will search for best HPs
# Kfold is integrated during the HP optimization 
# To use different ml architecture, define the model and parameters for HP tuning
# Use datasets to ensure training dataset is the same for both slearn and finetune workflow

# Split the data into train and test
def split_data(dataset, test_size, seed):
    # 80% train + valid, 20% test
    train_test = dataset['all'].train_test_split(test_size=test_size, seed=seed)
    # Split the 80% train + valid in half
    train_valid = train_test['train'].train_test_split(test_size=test_size, seed=seed)
    # gather everyone if you want to have a single DatasetDict
    dataset_splits = DatasetDict({
        'train': train_valid['train'],
        'test': train_test['test']})
    return dataset_splits

def truncate_train(dataset, seed, truncate=1.0):
    if truncate < 1.0:
        truncate_set = dataset['train'].train_test_split(test_size=truncate, seed=seed)
        dataset_splits = DatasetDict({
            'train': truncate_set['test'],
            'test': dataset['test']})
        return dataset_splits
    elif truncate == 1.0:
        return dataset
    else:
        print("Truncate size must be =< 1.0. Use full dataset.")
        return dataset

def load_features(inp_path, seqcol, ycol, seed, truncate=1.0):
    test_size = 0.2
    df = pd.read_csv(inp_path)
    df = df.reset_index(drop=True)
    # convert df into huggingface dataset format
    # for sklearn, only train and test is used
    if 'split' in df.columns:
        train_df = df[df['split'] == 'train']
        test_df = df[df['split'] == 'test']

        train_dataset = Dataset.from_pandas(train_df.drop(columns=['split']))
        test_dataset = Dataset.from_pandas(test_df.drop(columns=['split']))
        dataset_splits = DatasetDict({
            "train": train_dataset,
            "test": test_dataset
        })

    else:
        train_dataset = Dataset.from_pandas(df)
        dataset = DatasetDict({"all": train_dataset})
        dataset_splits = split_data(dataset, 0.2, seed)
        
    training_dataset = truncate_train(dataset_splits, seed, truncate)
    train = pd.DataFrame(training_dataset['train'])
    X_train = train.loc[:, seqcol:ycol].iloc[:, 1:-1]
    y_train = train[ycol]

    test = pd.DataFrame(training_dataset['test'])
    X_test = test.loc[:, seqcol:ycol].iloc[:, 1:-1]
    y_test = test[ycol]
    
    print(f"train: {len(X_train)}, test: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def spearmanr_metric(y_true, y_pred):
    r = spearmanr(a=y_true, b=y_pred)
    return r[0] 

def pearsonr_metric(y_true, y_pred):
    r = pearsonr(y_true, y_pred)
    return r[0] 

def set_scoring(metric):
    if metric == 'mse':
        return 'neg_root_mean_squared_error'
    if metric == 'acc':
        return 'accuracy'
    elif metric == 'spr':
        return make_scorer(spearmanr_metric)
    elif metric == 'per':
        return make_scorer(pearsonr_metric)
    else:
        print('wrong metric', metric)
        exit()

def train(X_train, X_test, y_train, y_test, seed, metric):
    clf = SVR()
    scoring=set_scoring(metric)
    clf.fit(X_train, y_train)
    # Evaluate model performance with test set
    y_pred = clf.predict(X_test)
    if metric == 'mse':
        score = root_mean_squared_error(y_test, y_pred)
    if metric == 'acc':
        score = accuracy_score(y_test, y_pred)
    elif metric == 'spr':
        score = spearmanr(y_test, y_pred)[0]
    elif metric == 'per':
        score = pearsonr(y_test, y_pred)[0]
    print('default score:', score)
    return (clf, score)

def optim(X_train, X_test, y_train, y_test, n_trials, seed, metric):
    clf = SVR()
    # revision 1 (for fluorescence)
    param_distributions = {
        'gamma': optuna.distributions.FloatDistribution(1e-4, 0.1, log=True),  # narrower than before
        'C': optuna.distributions.FloatDistribution(0.1, 10.0, log=True),      # avoid too small/large values
        'epsilon': optuna.distributions.FloatDistribution(0.01, 0.3, log=True) # smaller eps encourages learning
    }
        
    # Define the metric
    scoring=set_scoring(metric)
    
    # Define the Optuna search params
    optuna_search = optuna.integration.OptunaSearchCV(
        clf, param_distributions, n_trials=n_trials, random_state=seed, scoring=scoring,
    )

    # Train the model with the train set
    optuna_search.fit(X_train, y_train)
    
    try:
        for n, trial in enumerate(optuna_search.trials_):
            print(f'Trial {n}, score: {trial.values[0]}, params: {trial.params}')
    except Exception as e:
        print(f"Not able to print trial results. Error occurred: {e}.")
    # Print the best parameters found
    print(f"Best Optuna trial; {optuna_search.best_score_}; Best hyperparameters: {optuna_search.best_params_}")
    
    # Initialize model with best params and evaluate
    final_model = SVR(gamma=optuna_search.best_params_['gamma'], C=optuna_search.best_params_['C'], epsilon=optuna_search.best_params_['epsilon'])
    
    # Re-train the final model on the train set
    final_model.fit(X_train, y_train)

    # Evaluate model performance with test set
    y_pred = final_model.predict(X_test)    
    if metric == 'mse':
        score = root_mean_squared_error(y_test, y_pred)
    if metric == 'acc':
        score = accuracy_score(y_test, y_pred)
    elif metric == 'spr':
        score = spearmanr(y_test, y_pred)[0]
    elif metric == 'per':
        score = pearsonr(y_test, y_pred)[0]
    print('optim score:', score)
    return (final_model, score)

def main(args):
    orig_stdout = sys.stdout
    f = open(f"{args.out_path}.log", "w")
    sys.stdout = f
    start = time.time()

    # Step 1: Load data
    # Whether to load pre-compiled data splits (directory with _train/_test) or to generate data splits (csv input file)
    # Data splits will be automatically detected and created if not available
    X_train, X_test, y_train, y_test = load_features(args.inp_path, args.seqcol, args.ycol, args.seed, args.truncate)
    scaler = StandardScaler()
    # only fit on X_train (not performed on test set to ensure the dataset remain unseen)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 2: Perform hyperparameter tuning and model training
    if args.mode == 'default':
        final_model, score = train(X_train_scaled, X_test_scaled, y_train, y_test, args.seed, args.metric)
        print('test score:', score)
    elif args.mode == 'optim':
        default_model, default_score = train(X_train_scaled, X_test_scaled, y_train, y_test, args.seed, args.metric)
        final_model, optim_score = optim(X_train_scaled, X_test_scaled, y_train, y_test, args.n_trials, args.seed, args.metric)
        if (optim_score > default_score):
            print('test score:', optim_score)
        else:
            print('test score:', default_score)
    else:
        print('Mode of training was not defined (default or optim)')
    # Step 3: Save model for further use on prediction
    if args.save_model:
        pickle.dump(final_model, open(args.out_path + ".model.pickle", 'wb'))

    end = time.time()
    print("time taken: ", (end-start), "sec.")
    
    sys.stdout = orig_stdout
    f.close()

def my_bool(s): return s != 'False'
msg = "Perform supervised learning (hyperparameter tuning using grid search and model final training) on a given dataset using SVR for a regression task."
parser = argparse.ArgumentParser(description=msg)
# Adding optional argument (otherwise use default)
# Arguments to compute the features
parser.add_argument("-mode", "--mode", help = "Mode of training: default params 'default' or through HPO 'optim'", type=str, default="default")
parser.add_argument("-i", "--inp_path", help = "Path to input CSV containing the features and labels", type=str, required=True)
parser.add_argument("-o", "--out_path", help = "Path to log files", type=str, required=True)
parser.add_argument("-seqcol", "--seqcol", help = "Column describing the sequence", type=str, default='sequence')
parser.add_argument("-ycol", "--ycol", help = "Column describing the score/y values", type=str, default='label')
parser.add_argument("-t", "--truncate", help = "Truncate size (applied on train set)", type=float, default=1.0)
# Arguments for model training and optimization
parser.add_argument("-s", "--seed", help = "Random seed to use for cross-validation", type=int, default=42)
parser.add_argument("-m", "--metric", help = "Metric to use for evaluation, available: spr, per, mse, acc", type=str, default='spr')
parser.add_argument("-nt", "--n_trials", help = "Number of Optuna trials for optimization", type=int, default=30)
# Other arguments
parser.add_argument("-save", "--save_model", help = "Whether to save the model for further use", default=False, type=my_bool)
# Read arguments from command line
args = parser.parse_args()
main(args)

#python scripts/slearn.py -i ds/aafeat_features.csv -o aafeat_precomputed -p True -mode aafeat -f ms_whim_scores -seqcol seq -ycol label -m spr -nt 2
#python scripts/slearn.py -i data/fluorescence.csv -o aafeat_notprecomputed -p False -mode aafeat -f ms_whim_scores -seqcol seq -ycol label -m spr -nt 2
