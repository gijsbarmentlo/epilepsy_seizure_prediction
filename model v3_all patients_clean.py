# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 18:14:44 2021

@author: gijsb
"""

#%% Imports

import logging

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, roc_curve

from sklearn.metrics import make_scorer
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.model_selection import StratifiedKFold

from utils import compute_metrics, auc_patient_cv

import warnings
warnings.filterwarnings("ignore")



#%% Load features and create train and test datasets

# Load data from file

X_pat = {}
y_pat = {}

for p in [1,2,3]:
    X_pat[f'pat{p}'] = np.load(f'neurovista_X_train_pat{p}.npy').astype('float32')
    X_pat[f'pat{p}'] = np.nan_to_num(X_pat[f'pat{p}'])
    
    y_pat[f'pat{p}'] = np.load(f'neurovista_y_train_pat{p}.npy').astype('float32')
    y_pat[f'pat{p}'] = np.nan_to_num(y_pat[f'pat{p}'])
    
logging.debug('X and y loaded into dictionary')

# Assign data to train and test sets

## Per patient split to guarantee all subsets have equal proportions of data from each patient

X_train_dict = {}
X_test_dict = {}
y_train_dict = {}
y_test_dict = {}

for p in [1, 2, 3]:
    X_train_dict[f'pat{p}'], X_test_dict[f'pat{p}'], y_train_dict[f'pat{p}'], y_test_dict[f'pat{p}'] = train_test_split(X_pat[f'pat{p}'], y_pat[f'pat{p}'], test_size=0.4, random_state=42)

X_train = np.vstack(tuple(X_train_dict.values()))
X_test = np.vstack(tuple(X_test_dict.values()))
y_train = np.vstack(tuple(y_train_dict.values()))
y_test = np.vstack(tuple(y_test_dict.values()))

#%% Test basic models to choose which ones to tune further


from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm

# Create basic classifiers and evaluate without hyperparam tuning

classifiers = {'ExtraTrees' : ExtraTreesClassifier(n_jobs = -1), 
               'Kneighbours' : KNeighborsClassifier(3, n_jobs = -1),
               'Gaussian process' : GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs = -1),
               'Decision tree' : DecisionTreeClassifier(),
               'Random forest' : RandomForestClassifier(n_estimators=500, n_jobs = -1),
               'Neural network' : MLPClassifier(alpha=1, max_iter=1000),
               'Adaboost' : AdaBoostClassifier(),
               'Gaussain NB' : GaussianNB(),
               'Quadratic discriminant' : QuadraticDiscriminantAnalysis(),
               'Hist Gradient Boosting Classifier (LGBM-like)' : HistGradientBoostingClassifier(),
               'LGBM' : lightgbm.LGBMClassifier(n_estimators = 500, objective = 'binary')}

performance_dict = {}

for clf_name, clf in classifiers.items():
    # carefull per patient split
    clf.fit(X_train, y_train)
    metrics = compute_metrics(clf, X_test, y_test)
    print(f"patient split : {metrics} \n")
    performance_dict[f'{clf_name}_pat'] = metrics
    
for clf_name, clf in classifiers.items():
    # Train on 2 patients test on the other
    auc_dict = auc_patient_cv(clf, X_pat, y_pat)
    print(f'train on 2 out of 3 : {auc_dict}')
    
# LGBM and extratrees perform best

#%% Simple LGBM

# create and fit 
import lightgbm
basic_lgbm = lightgbm.LGBMClassifier(n_estimators = 500, objective = 'binary', class_weight = 'balanced')
basic_lgbm.fit(X_train, y_train, eval_metric = 'auc', verbose = 2)


# Compute metrics and plot ROC
metrics_dict = compute_metrics(basic_lgbm, X_test, y_test)
print(f'extratrees train on 2/3rd of each patient : {metrics_dict}')

y_pred_proba = basic_lgbm.predict_proba(X_test)[:, [1]]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - Basic LGBM Classifier')
plt.plot(fpr, tpr)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

del basic_lgbm

#%% Simple ExtraTrees
    
basic_extratrees = ExtraTreesClassifier()
basic_extratrees.fit(X_train, y_train)

# Evaluate performance before tuning
metrics_dict = compute_metrics(basic_extratrees, X_test, y_test)

y_pred_proba = basic_extratrees.predict_proba(X_test)[:, [1]]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - Basic ExtraTrees Classifier')
plt.plot(fpr, tpr)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

print (f'without hyperparameter tuning, the ExtraTreesClassifier performs as follows : {metrics_dict}')

del basic_extratrees


#%% ExtraTrees - hyperparameter tuning with genetic algo


# Create model 
extra_trees_clf = ExtraTreesClassifier(n_jobs = -1)

# Optimise hyperparameters

scorer_object = make_scorer(roc_auc_score, greater_is_better = True, needs_proba = True)

distributions = dict(criterion = ['gini', 'entropy'],
                     n_estimators = range(200, 8000), 
                     max_depth = [50, 100, 150, 200, 1000, None],
                     min_samples_split = range(20, 100),
                     min_samples_leaf = range(10, 200),
                     max_features = ['auto', 'sqrt', 'log2'],
                     max_leaf_nodes = [10, 100, 500, 1000, None],
                     min_impurity_decrease = [0, 0.01, 0.1],
                     bootstrap = [False, True],
                     class_weight = ['balanced', 'balanced_subsample', None],
                     ccp_alpha = np.linspace(0, 0.08, 20) ,
                     max_samples = np.linspace(0.001, 0.5, 20).astype('float'))


search_xt = EvolutionaryAlgorithmSearchCV(extra_trees_clf, 
                                    distributions, 
                                    population_size = 100,
                                    gene_mutation_prob = 0.10,
                                    gene_crossover_prob = 0.5,
                                    tournament_size = 3,
                                    generations_number = 6,
                                    cv = StratifiedKFold(n_splits = 3),
                                    verbose = 1,
                                    scoring = scorer_object,
                                    n_jobs = 1)
                                    #pmap = pool.map

search_xt.fit(X_train, y_train) 
print(f'best parameters found in search are {search_xt.best_params_}')

best_xt = search_xt.best_estimator_ 
metrics_dict = compute_metrics(best_xt, X_test, y_test)
print(f'model trained on all patients has following performance metrics after hyperparam tuning: {metrics_dict}')

#%% LGBM - hyperparameter tuning with genetic algo

# Instantiate lgbm with basic hyperparameters that speed up training 
# so we can cover a larger space of hyperparameters
# Since performance is maxed out i am more worried abvout overfitting so i will
# use kwargs that reduce overfitting

lgbm_clf = lightgbm.LGBMClassifier(objective = 'binary', n_jobs = -1, 
                                   feature_pre_filter = True, 
                                   metric = 'auc', 
                                   force_col_wise = True)

scorer_object = make_scorer(roc_auc_score, 
                            greater_is_better = True, 
                            needs_proba = True)

distributions = dict(num_leaves = [100, 300, 500],
                    min_data_in_leaf = [5, 10, 30, 100, 200],
                    bagging_fraction = [0.9, 0.8, 0.7, 0.6, 0.5],
                    bagging_freq = [3, 5, 10, 20],
                    feature_fraction = [1, 0.9, 0.8, 0.7, 0.6, 0.5],
                    max_depth = [10, 50, 100, 200, 500],
                    num_iterations = [50, 100, 250, 500],
                    is_unbalance = [True, False],
                    extra_trees = [True, False],
                    lambda_l1 = [0.0, 0.1, 0.5])

search_lgbm = EvolutionaryAlgorithmSearchCV(lgbm_clf, 
                                    distributions, 
                                    population_size = 50,
                                    gene_mutation_prob = 0.10,
                                    gene_crossover_prob = 0.5,
                                    tournament_size = 3,
                                    generations_number = 3,
                                    cv = StratifiedKFold(n_splits = 3),
                                    verbose = 1,
                                    scoring = scorer_object,
                                    n_jobs = 1)

search_lgbm.fit(X_train, y_train)
print(f'best parameters found in search are {search_lgbm.best_params_}')

best_lgbm = search_lgbm.best_estimator_ 
metrics_dict = compute_metrics(best_lgbm, X_test, y_test)
print(f'model trained on all patients has following performance metrics after hyperparam tuning: {metrics_dict}')

# best parameters found in search are {'num_leaves': 500, 'min_data_in_leaf': 30, 'bagging_fraction': 0.9, 'bagging_freq': 3, 'feature_fraction': 0.7, 'max_depth': 10, 'num_iterations': 500, 'is_unbalance': True, 'extra_trees': False, 'lambda_l1': 0.0}
# model trained on all patients has following performance metrics after hyperparam tuning: {'auc': 0.9967769030137162, 'f1score': 0.9208103130755064, 'accuracy': 0.8689719670377413, 'balanced_accuracy': 0.8676199317603654, 'recall': 0.8710801393728222}

#%% LGBM avoiding overfitting

lgbm_clf = lightgbm.LGBMClassifier(n_estimators = 500, objective = 'binary', 
                                   num_leaves = 100, min_data_in_leaf = 30, bagging_fraction = 0.8, 
                                   bagging_freq = 5, feature_fraction = 0.6, max_depth = 500,  
                                   lambda_l1 = 0.1) #is_unbalance = True,

lgbm_clf.fit(X_train, y_train, early_stopping_rounds=5, eval_set = [(X_test, y_test)], 
             eval_metric = 'auc', verbose = 2)

# Compute metrics and plot ROC
metrics_dict = compute_metrics(lgbm_clf, X_test, y_test)
print(f'extratrees train on 2/3rd of each patient : {metrics_dict}')

y_pred_proba = lgbm_clf.predict_proba(X_test)[:, [1]]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - Extra Trees Classifier')
plt.plot(fpr, tpr)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()