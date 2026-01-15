import os, sys, argparse
import pandas as pd
import numpy as np
import peptides
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_validate, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import make_scorer
from scipy.stats import spearmanr, pearsonr
import warnings
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.metrics import mean_squared_error, r2_score
import warnings
import time

# sigmoid
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

# create a new 'score' column to decsribe the intensity strength
def sigmoid_function(df, ycol):
    thsh = 1.0

    df[ycol] = (
        sigmoid(df['Intensity'] - thsh)
        * sigmoid(df['Change'] - thsh)
        - 1.0
    )
    return df

# extract amino acid features from a given descriptor
def compute_features(df, seqcol, feature):
    # Available features: available: blosum_indices fasgai_vectors ms_whim_scores protfp_descriptors st_scales t_scales vhse_scale z_scales
    # and many more (refer to peptides package in github: https://github.com/althonos/peptides.py)

    # Get features from a descriptor
    data1=[[list(getattr(peptides.Peptide(a_a), feature)()) for a_a in list(seq)] for seq in df[seqcol]]
    data2=[[s for j in k for s in j] for k in data1]

    # Retrieve features (X) and target values (y)
    X = pd.DataFrame(data2)
    
    return X

# Define metric loading function
def pearsonr_metric(y_true, y_pred):
    r = pearsonr(x=y_true, y=y_pred)
    return r[0] 

def spearmanr_metric(y_true, y_pred):
    r = spearmanr(a=y_true, b=y_pred)
    return r[0] 

def set_scoring(metric):
    if metric == 'r2':
        return 'r2'
    elif metric == 'rmse':
        return 'neg_root_mean_squared_error'
    elif metric == 'pearson':
        return make_scorer(pearsonr_metric)
    elif metric == 'spearman':
        return make_scorer(spearmanr_metric)
    else:
        print('wrong metric', metric)
        exit()

def model_selection(default_model, X, y, metric, RANDOM_STATE, param_grid=None):
    # Define params
    N_SPLITS=5

    # Load metric
    scoring = set_scoring(metric)

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Define outer CV
    outer_cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignores all warnings
        
        if param_grid: # run optimization if param_grid provided
            # Only use nested (inner and outer) CV when optimization is performed
            inner_cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            model = GridSearchCV(estimator=default_model, param_grid=param_grid, cv=inner_cv, scoring=scoring, verbose=1) # inner CV
            # Perform cross-validation with Kfold
            res = cross_validate(estimator=model, X=X, y=y, cv=outer_cv, scoring=scoring, return_estimator=True, verbose=1) # outer CV
            
            if param_grid:
                print('cv scores and parameters:')
                for i in range(N_SPLITS):
                    print(res['test_score'][i], res['estimator'][i].best_params_)
        
        
        else:
            model = default_model
            # Perform cross-validation with Kfold
            res = cross_validate(estimator=model, X=X, y=y, cv=outer_cv, scoring=scoring, return_estimator=True, verbose=1) # outer CV
            
            print('cv scores:')
            for i in range(N_SPLITS):
                print(res['test_score'][i])
        
        print('mean score:')
        print(res['test_score'].mean())

def model_construction(default_model, X, y, metric, RANDOM_STATE, param_grid=None):
    # Define params
    N_SPLITS=5

    # Load metric
    scoring = set_scoring(metric)

    # Scale the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Define outer CV
    outer_cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # ignores all warnings
        
        if param_grid: # run optimization if param_grid provided
            # Only use nested (inner and outer) CV when optimization is performed
            inner_cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
            model = GridSearchCV(estimator=default_model, param_grid=param_grid, cv=inner_cv, scoring=scoring, verbose=1) # inner CV
            
            # Cross validation will not be performed here
            # Fit the best hyperparamater to construct updated model
            model.fit(X=X, y=y)
            
            print('best parameter:', model.best_params_)
            print('best score:', model.best_score_)

            # get best estimator
            bsmodel = model.best_estimator_

        else:
            bsmodel = default_model
            
            # Fit the default hyperparamater to construct updated model
            bsmodel.fit(X=X, y=y)

            # Evaluate the prediction
            y_pred = bsmodel.predict(X)
            if metric == 'r2':
                val = r2_score(y, y_pred)
            elif metric == 'rmse':
                val = - np.sqrt(mean_squared_error(y, y_pred, squared=False))
            elif metric == 'pearson':
                val = pearsonr_metric(y, y_pred)
            elif metric == 'spearman':
                val = spearmanr_metric(y, y_pred)
            else:
                print('wrong metric', metric)
                exit()    
            
            print('default score:', val)

    return bsmodel, scaler

def model_prediction(pred_csv, ycol, model, scaler, metric):
    
    # Load prediction list
    df = pd.read_csv(pred_csv)

    # Define X and y here
    X = df.iloc[:, 1:-1]
    y = df[ycol]
    
    # Transform the features (do not perform fit_transform so that it remain unseen)
    # Scaler must be the same scaler used for training and hyperparameter optimization
    X_scaled = scaler.transform(X)
    
    # Make prediction
    y_pred = model.predict(X_scaled)

    # Evaluate the prediction
    if metric == 'r2':
        val = r2_score(y, y_pred)
    elif metric == 'rmse':
        val = - np.sqrt(mean_squared_error(y, y_pred, squared=False))
    elif metric == 'pearson':
        val = pearsonr_metric(y, y_pred)
    elif metric == 'spearman':
        val = spearmanr_metric(y, y_pred)
    else:
        print('wrong metric', metric)
        exit()
    
    print('score (actual vs. prediction):', val)

    # Create a dataframe and sort
    pred = pd.concat([df['Sequence'], df['Score'], pd.DataFrame(y_pred, columns=['Pred'])], axis=1)
    
    return (pred) 

# add more model and adjust grid here
def load_model_params(model_name, RANDOM_STATE):
    if model_name == 'svr':
        model = SVR()
        grid = {'gamma': [1e-5, 1e-4, 1e-3, 1e-2, 1e-1], 'C': [1e-2, 1e-1, 1e0, 1e1, 1e2], 'epsilon': [1e-4, 1e-3, 1e-2, 1e-1, 1e0]}
    elif model_name == 'linr':
        model = Lasso(max_iter=100000)
        grid = {'alpha': [1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1, 2, 5, 1e+1, 2e+1, 5e+1, 1e+2]}
    elif model_name == 'gpr':
        # try for GPR
        # n_restarts_optimizer = different random starting points
        # it controls how many times optimizer restarts when fitting the kernel hyperparameters
        # thus no params will be used here for optimization
        #model = GaussianProcessRegressor(n_restarts_optimizer=10, normalize_y=True, random_state=RANDOM_STATE)
        model = GaussianProcessRegressor(normalize_y=True, random_state=RANDOM_STATE)
        grid = {'n_restarts_optimizer': [5, 10, 15, 20]}
    return (model, grid)

def workflow(model_name, X, y, pred_csv, ycol, metric, seed, out_path, save_model=False):
    # Load model and parameter grid search
    print("Load model and parameter grid..")
    model, grid = load_model_params(model_name, seed)
    print("\n---- MODEL SELECTION ----\n")
    print("Default Hyperparameters..")
    default = model_selection(model, X, y, metric, seed, param_grid=None)
    print("\nHyperparameter optimization with GridSearchCV..")
    optim = model_selection(model, X, y, metric, seed, grid)
    print("\n---- MODEL CONSTRUCTION ----\n")
    print("Default Hyperparameters..")
    df_model, df_scaler = model_construction(model, X, y, metric, seed)
    print("\nHyperparameter optimization with GridSearchCV..")
    bs_model, bs_scaler = model_construction(model, X, y, metric, seed, grid)
    print("\n---- MODEL PREDICTION ---- \n")
    if pred_csv !=None:
        results = model_prediction(pred_csv, ycol, bs_model, bs_scaler, metric)
        results_sorted = results.sort_values(by='Pred', ascending=False)
        results_sorted.to_csv(out_path+".csv")
        if save_model:
            pickle.dump(bs_model, open(out_path + ".model.pickle", 'wb'))
            pickle.dump(bs_scaler, open(out_path + ".scaler.pickle", 'wb'))
    else:
        print("No prediction list supplied!")

def main(args):
    orig_stdout = sys.stdout
    f = open(f"{args.out_path}.log", "w")
    sys.stdout = f
    start = time.time()

    # Load dataset
    # Default sequence column = 'Sequence'
    df = pd.read_csv(args.inp_csv)

    # Create a new 'Score' column to decsribe the intensity strength
    # Default label column = 'Score'
    df = sigmoid_function(df, args.ycol)

    # Get features for all sequences in the dataset (X and y)
    X = compute_features(df, args.seqcol, args.feature) 
    y = df[args.ycol]

    # Load model and parameter grid search
    model, grid = load_model_params(args.model_name, args.seed)

    if args.mode == 'ms':
        # Use default configuration
        default = model_selection(model, X, y, args.metric, args.seed, param_grid=None)
        # Optimize SVR hyperparameters
        optim = model_selection(model, X, y, args.metric, args.seed, grid)
    elif args.mode == 'mc':
        # Use default configuration
        df_model, df_scaler = model_construction(model, X, y, args.metric, args.seed)
        # Optimize SVR hyperparameters
        bs_model, bs_scaler = model_construction(model, X, y, args.metric, args.seed, grid)
    elif args.mode == 'mp':
        if args.pred_csv != None:
            bs_model, bs_scaler = model_construction(model, X, y, args.metric, args.seed, grid)
            results = model_prediction(args.pred_csv, ycol, bs_model, bs_scaler, metric)
            results_sorted = results.sort_values(by='Pred', ascending=False)
            results_sorted.to_csv(args.out_path+".csv")
            if args.save_model:
                pickle.dump(bs_model, open(args.out_path + ".model.pickle", 'wb'))
                pickle.dump(bs_scaler, open(args.out_path + ".scaler.pickle", 'wb'))
        else:
            print("No prediction list supplied!")
    elif args.mode == 'workflow':
        workflow(args.model_name, X, y, args.pred_csv, args.ycol, args.metric, args.seed, args.out_path, args.save_model)

    else:
        print("Select between three models: ms, mc, mp")

    end = time.time()
    print("time taken: ", (end-start), "sec.")
    
    sys.stdout = orig_stdout
    f.close()

"""
inp_csv = "data/umetsu/Umetsu_GFP.csv"
feature="t_scales"
seqcol="Sequence"
ycol="Score"
df = pd.read_csv(inp_csv)
print(df.head())
"""


def my_bool(s): return s != 'False'
msg = "Machine learning with GFP dataset"
parser = argparse.ArgumentParser(description=msg)
# Adding optional argument (otherwise use default)
# Arguments to compute the features
parser.add_argument("-mode", "--mode", help = "Mode of training: ms, mc, mp, or workflow to run all steps", type=str, default="default")
parser.add_argument("-i", "--inp_csv", help = "Path to input CSV containing the features and labels", type=str, required=True)
parser.add_argument("-p", "--pred_csv", help = "Path to input CSV containing the features for prediction", type=str, default=None)
parser.add_argument("-o", "--out_path", help = "Path to log and output files", type=str, required=True)
parser.add_argument("-feature", "--feature", help = "Amino acid descriptor to generate training features", type=str, default='t_scales')
parser.add_argument("-seqcol", "--seqcol", help = "Column describing the sequence", type=str, default='Sequence')
parser.add_argument("-ycol", "--ycol", help = "Column describing the score/y values", type=str, default='Score')
# Arguments for model training and optimization
parser.add_argument("-model", "--model_name", help = "Model architecture to use: 'gpr', 'svr' or 'linr'", type=str, required=True)
parser.add_argument("-s", "--seed", help = "Random seed to use for cross-validation", type=int, default=42)
parser.add_argument("-m", "--metric", help = "Metric to use for evaluation, available: spr, per, mse, acc", type=str, default='spearman')
# Other arguments
parser.add_argument("-save", "--save_model", help = "Whether to save the model for further use", default=False, type=my_bool)
# Read arguments from command line
args = parser.parse_args()
main(args)

