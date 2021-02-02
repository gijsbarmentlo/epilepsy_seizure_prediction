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
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
import sklearn.metrics


#%% Helper functions


def compute_metrics(clf, X_test, y_test):
    """
    return a dict containing auc, f1score, accuracy, balanced_accuracy, recall
    """
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, [1]]
    
    auc = roc_auc_score(y_test, y_pred_proba)
    f1score = f1_score(y_test, y_pred)
    accuracy = sklearn.metrics.average_precision_score(y_test, y_pred)
    balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_test, y_pred, adjusted = True)
    recall = sklearn.metrics.recall_score(y_test, y_pred)
    
    metrics_dict = {}
    metrics_dict['auc'] = auc
    metrics_dict['f1score'] = f1score
    metrics_dict['accuracy'] = accuracy
    metrics_dict['balanced_accuracy'] = balanced_accuracy
    metrics_dict['recall'] = recall
    
    return metrics_dict

def compute_metrics_threshold(clf, X_test, y_test, threshold):
    """
    return a dict containing auc, f1score, accuracy, balanced_accuracy, recall
    """
    y_pred_proba = clf.predict_proba(X_test)[:, [1]]
    y_pred = (y_pred_proba > threshold).astype('int')
    
    auc = roc_auc_score(y_test, y_pred_proba)
    f1score = f1_score(y_test, y_pred)
    accuracy = sklearn.metrics.average_precision_score(y_test, y_pred)
    balanced_accuracy = sklearn.metrics.balanced_accuracy_score(y_test, y_pred, adjusted = True)
    recall = sklearn.metrics.recall_score(y_test, y_pred)
    
    metrics_dict = {}
    metrics_dict['auc'] = auc
    metrics_dict['f1score'] = f1score
    metrics_dict['accuracy'] = accuracy
    metrics_dict['balanced_accuracy'] = balanced_accuracy
    metrics_dict['recall'] = recall
    
    return metrics_dict

#%% Create train and test datasets

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

X = np.vstack(tuple(X_pat.values()))
y = np.vstack(tuple(y_pat.values()))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


#%% Model for all patients with cross patient test/train split - first fit

# Create basic classifiers

basic_clf = ExtraTreesClassifier()
basic_clf.fit(X_train, y_train)


# Evaluate performance before tuning
metrics_dict = compute_metrics(basic_clf, X_test, y_test)

y_pred_proba = basic_clf.predict_proba(X_test)[:, [1]]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)

print (f'without hyperparameter tuning, the ExtraTreesClassifier performs as follows : {metrics_dict}')



#%% find appropriate alpha range for pruning

# to avoid overfitting we will use minimal cost-complexity tuning https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning

from sklearn.tree import DecisionTreeClassifier
alpha_tree_clf = DecisionTreeClassifier()
path = alpha_tree_clf.cost_complexity_pruning_path(X, y)
ccp_alphas, impurities = path.ccp_alphas, path.impurities

fig, ax = plt.subplots()
ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
ax.set_xlabel("effective alpha")
ax.set_ylabel("total impurity of leaves")
ax.set_title("Total Impurity vs effective alpha for training set")


#%% Model for all patients - hyperparameter tuning with genetic algo


from sklearn.metrics import make_scorer
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.model_selection import StratifiedKFold
from multiprocessing import Pool

# Create model 
extra_trees_clf = ExtraTreesClassifier(n_jobs = -1, bootstrap = False, max_samples = 0.2)

# Optimise hyperparameters

scorer_object = make_scorer(roc_auc_score, greater_is_better = True, needs_proba = True)

distributions = dict(criterion = ['gini', 'entropy'],
                     n_estimators = range(10, 1000), 
                     max_depth = [1, 10, 100, 'None'],
                     min_samples_split = range(2, 50),
                     min_samples_leaf = range(1, 200),
                     max_features = ['Auto', 'sqrt', 'log2'],
                     max_leaf_nodes = [10, 100, 1000, 'None'],
                     min_impurity_decrease = [0, 0.01, 0.25, 1, 10],
                     bootstrap = [False, True],
                     class_weight = ['balanced', 'balanced_subsample', 'None'],
                     ccp_alpha = np.linspace(0, 0.013, 13) ,
                     max_samples = np.linspace(0, 1, 20))

pool = Pool(4)

clf = EvolutionaryAlgorithmSearchCV(extra_trees_clf, 
                                    distributions, 
                                    population_size=50,
                                    gene_mutation_prob = 0.10,
                                    gene_crossover_prob = 0.5,
                                    tournament_size = 3,
                                    generations_number = 5,
                                    cv = StratifiedKFold(n_splits = 3),
                                    verbose = 1,
                                    scoring = scorer_object,
                                    pmap = pool.map)
                                    #n_jobs = 1

search = clf.fit(X_train, y_train)
print(f'best parameters found in search are {search.best_params_}')

#%%

best_clf = ExtraTreesClassifier(n_estimators = 837, max_depth = 38, min_samples_leaf = 1, min_impurity_decrease = 0, criterion = 'entropy', bootstrap = False, min_samples_split = 2, class_weight = 'balanced')

#best_clf = ExtraTreesClassifier(**clf.best_params_)
best_clf = best_clf.fit(X_train, y_train)
metrics_dict = compute_metrics(best_clf, X_test, y_test)
print(f'model trained on patients 2 has following performance metrics after hyperparam tuning: {metrics_dict}')

#import pickle
#filename = 'neurovista_model_pat3_31jan_hyperparam_genetic.sav'
#pickle.dump(best_clf, open(filename, 'wb'))

#%% evaluate
import pickle

with open(r"neurovista_model_pat3_31jan_hyperparam_genetic.sav", "rb") as input_file:
    best_clf = pickle.load(input_file)

y_pred_proba = best_clf.predict_proba(X_test)[:, [1]]


import pandas as pd
from sklearn.metrics import roc_curve

for threshold in np.linspace(0, 1, 50):
    y_pred = (y_pred_proba > threshold).astype('int')
    recall = sklearn.metrics.recall_score(y_test, y_pred)
    print(f'for theshold at {threshold} recall = {recall}')
    

fpr, tpr, threshold = roc_curve(y_test, y_pred_proba)

import matplotlib.pyplot as plt
plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - Extra Trees Classifier')
plt.plot(fpr, tpr)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#Ù From ROC we can choose threshold at 0.24 to 80% precision
threshold = 0.24
y_pred = (y_pred_proba > threshold).astype('int').tolist()
y_test_list = y_test.tolist()
metrics_dict = compute_metrics_threshold(best_clf, X_test, y_test, threshold)
print(f'after hyperparam tuning and putting threshhold at {threshold} we have these performance metrics {metrics_dict}')

threshold = 0.3
y_pred = (y_pred_proba > threshold).astype('int').tolist()
y_test_list = y_test.tolist()
metrics_dict = compute_metrics_threshold(best_clf, X_test, y_test, threshold)
print(f'after hyperparam tuning and putting threshhold at {threshold} we have these performance metrics {metrics_dict}')

# precision = sklearn.metrics.precision_score(y_test, y_pred)
# print(precision)

# TODO make a dataframe with the metrics per  threshold so i can plot it nice


#metrics_dict = compute_metrics(best_clf, X_test, y_test)
#print(f'model trained on patients 2 and 3 has following performance metrics after hyperparam tuning: {metrics_dict}')


#%% saving model to file

#import pickle
#filename = 'neurovista_pat2and3_30jan.sav'
#pickle.dump(clf, open(filename, 'wb'))


#%% EDA

import pandas as pd

X_df = pd.DataFrame(X)
print(X_df.describe())

y_df = pd.DataFrame(y)
print(y_df.describe())