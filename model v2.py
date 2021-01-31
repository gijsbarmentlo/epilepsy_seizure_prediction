# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 18:14:44 2021

@author: gijsb
"""
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, f1_score
import sklearn.metrics
import logging

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

#%% Load data from file into dict

X_pat = {}
y_pat = {}

for p in [1,2,3]:
    X_pat[f'pat{p}'] = np.load(f'neurovista_X_train_pat{p}.npy').astype('float32')
    X_pat[f'pat{p}'] = np.nan_to_num(X_pat[f'pat{p}'])
    
    y_pat[f'pat{p}'] = np.load(f'neurovista_y_train_pat{p}.npy').astype('float32')
    y_pat[f'pat{p}'] = np.nan_to_num(y_pat[f'pat{p}'])
    
logging.debug('X and y loaded into dictionary')

#%% patient specific models

for p in [1, 2, 3]:
    X = X_pat[f'pat{p}']
    X = np.nan_to_num(X)
    
    y = y_pat[f'pat{p}']
    y = np.nan_to_num(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    
    #fit model
    clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    
    # Evaluate model    
    metrics_dict = compute_metrics(clf, X_test, y_test)
        
    print(f'model for patient {p} has following performance metrics: {metrics_dict} \n')

#%% Model for all patients with cross patient test/train split



# Assign data to train and test sets
#X = np.vstack((X_pat['pat3'], X_pat['pat2'], X_pat['pat1']))
#y = np.vstack((y_pat['pat3'], y_pat['pat2'], y_pat['pat1']))

X = np.vstack(tuple(X_pat.values()))
y = np.vstack(tuple(y_pat.values()))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

clf = ExtraTreesClassifier(n_estimators=100, random_state=42, max_features = 'auto', bootstrap = True, class_weight= 'balanced_subsample')
clf.fit(X_train, y_train)

metrics_dict = compute_metrics(clf, X_test, y_test)

#fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba, pos_label=1)
#auc_2 = auc(fpr, tpr)

print (metrics_dict)


#print(f'model for both patients 2 and 3 performance metrics : \n AUC : {auc} \n f1_score = {f1score} \n accuracy : {accuracy} \n  ')


#%% Train model on two patients and test on another

#TODO rewirte with x_pat dictionnary

# Assign data to train and test sets
X_train = np.vstack((X_pat['pat3'], X_pat['pat2']))
y_train = np.vstack((y_pat['pat3'], y_pat['pat2']))

#X_train = X_pat['pat3']
#y_train = y_pat['pat3']

X_test = np.nan_to_num(X_pat['pat1'].astype('float32'))
y_test = np.nan_to_num(y_pat['pat1'].astype('float32'))

# Fit model on train data (ie. patients 2 and 3)
cross_patient_clf = ExtraTreesClassifier(n_estimators=100, random_state=42, max_features = 'auto', bootstrap = True, class_weight= 'balanced_subsample')
cross_patient_clf.fit(X_train, y_train)

# Evaluate model on patient 1 to see if the model generalises across patients
auc = roc_auc_score(y_test, cross_patient_clf.predict_proba(X_test)[:, 1])
f1score = f1_score(y_test, cross_patient_clf.predict(X_test))

print(f'model trained on two patients and tested on another has AUC : {auc} and f1_score = {f1score}')



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

#%% hyperparameter tuning

from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import make_scorer
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.model_selection import StratifiedKFold
from multiprocessing import Pool

# Create train and test sets from one or many patients

#X = np.vstack((X_pat['pat3'], X_pat['pat2']))
#y = np.vstack((y_pat['pat3'], y_pat['pat2']))

X = X_pat['pat3']
y = y_pat['pat3']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)


# Create model and evaluate before optimisation
extra_trees_clf = ExtraTreesClassifier()
extra_trees_clf.fit(X_train, y_train)
metrics_dict = compute_metrics(extra_trees_clf, X_test, y_test)
print(f'model trained on patients 2 has following performance metrics before hyperparam tuning: {metrics_dict}')


# Optimise hyperparams
scorer_object = make_scorer(roc_auc_score, greater_is_better = True, needs_proba = True)

distributions = dict(n_estimators = range(10, 1000), 
                     max_depth = range(5, 40),  
                     min_samples_leaf = [1,2,5], 
                     min_impurity_decrease = [0, 0.1], 
                     criterion = ['gini', 'entropy'],
                     bootstrap = [False, True],
                     min_samples_split = range(2, 10),
                     class_weight = ['balanced', 'balanced_subsample', None])

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

#{'n_estimators': 837, 'max_depth': 38, 'min_samples_leaf': 1, 'min_impurity_decrease': 0, 'criterion': 'entropy', 'bootstrap': False, 'min_samples_split': 2, 'class_weight': 'balanced'}

best_clf = ExtraTreesClassifier(**clf.best_params_)
best_clf = best_clf.fit(X_train, y_train)
metrics_dict = compute_metrics(best_clf, X_test, y_test)
print(f'model trained on patients 2 has following performance metrics after hyperparam tuning: {metrics_dict}')

import pickle
filename = 'neurovista_model_pat3_31jan_hyperparam_genetic.sav'
pickle.dump(best_clf, open(filename, 'wb'))

#%% evaluate

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

precision = sklearn.metrics.precision_score(y_test, y_pred)
print(precision)

metrics_dict = compute_metrics_threshold(best_clf, X_test, y_test, threshold)

print(f'after hyperparam tuning and putting threshhold at {threshold} we have these performance metrics {metrics_dict}')

#metrics_dict = compute_metrics(best_clf, X_test, y_test)
#print(f'model trained on patients 2 and 3 has following performance metrics after hyperparam tuning: {metrics_dict}')
