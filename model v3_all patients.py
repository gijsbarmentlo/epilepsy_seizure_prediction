# -*- coding: utf-8 -*-
"""
Created on Fri Jan 29 18:14:44 2021

@author: gijsb
"""

#%% Imports


import logging
import pickle

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import roc_auc_score, f1_score, roc_curve
import sklearn.metrics

import warnings
warnings.filterwarnings("ignore")

#%% Helper functions

# TODO add kfold and remove threshold commpute

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

def auc_patient_cv(clf, X_pat, y_pat):
    
    auc_dict = {}
    
    # A - train = 1 & 2 ; test = 3
    X_train = np.vstack((X_pat['pat1'], X_pat['pat2']))
    y_train = np.vstack((y_pat['pat1'], y_pat['pat2']))
    X_test = X_pat['pat3']
    y_test = y_pat['pat3']
    
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, [1]]
    auc_dict['train 1_2'] = roc_auc_score(y_test, y_pred_proba)
    
    # B - train = 1 & 3 ; test = 2
    X_train = np.vstack((X_pat['pat1'], X_pat['pat3']))
    y_train = np.vstack((y_pat['pat1'], y_pat['pat3']))
    X_test = X_pat['pat2']
    y_test = y_pat['pat2']
    
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, [1]]
    auc_dict['train 1_3'] = roc_auc_score(y_test, y_pred_proba)    

    # C - train = 2 & 3 ; test = 1
    X_train = np.vstack((X_pat['pat2'], X_pat['pat3']))
    y_train = np.vstack((y_pat['pat2'], y_pat['pat3']))
    X_test = X_pat['pat1']
    y_test = y_pat['pat1']
    
    clf.fit(X_train, y_train)
    y_pred_proba = clf.predict_proba(X_test)[:, [1]]
    auc_dict['train 2_3'] = roc_auc_score(y_test, y_pred_proba)     
    
    return auc_dict

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

## Random split

X = np.vstack(tuple(X_pat.values()))
y = np.vstack(tuple(y_pat.values()))

X_train_rand, X_test_rand, y_train_rand, y_test_rand = train_test_split(X, y, test_size=0.33, random_state=42)

## Per patient split to guarantee all subsets have equal proportions of data from each patient

X_train_dict = {}
X_test_dict = {}
y_train_dict = {}
y_test_dict = {}

for p in [1, 2, 3]:
    X_train_dict[f'pat{p}'], X_test_dict[f'pat{p}'], y_train_dict[f'pat{p}'], y_test_dict[f'pat{p}'] = train_test_split(X_pat[f'pat{p}'], y_pat[f'pat{p}'], test_size=0.4, random_state=42)

X_train_pat = np.vstack(tuple(X_train_dict.values()))
X_test_pat = np.vstack(tuple(X_test_dict.values()))
y_train_pat = np.vstack(tuple(y_train_dict.values()))
y_test_pat = np.vstack(tuple(y_test_dict.values()))

#%% Model for all patients with cross patient test/train split - first fit

# Create basic classifiers

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import RidgeClassifier
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
import lightgbm



classifiers = {'ExtraTrees' : ExtraTreesClassifier(n_jobs = -1), 
               'Kneighbours' : KNeighborsClassifier(3, n_jobs = -1),
#               'Gaussian process' : GaussianProcessClassifier(1.0 * RBF(1.0), n_jobs = -1),
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
    # random split
    clf.fit(X_train_rand, y_train_rand)
    metrics = compute_metrics(clf, X_test_rand, y_test_rand)
    print(f'{clf_name}')
    print(f"random split : {metrics}")
    performance_dict[f'{clf_name}_rand'] = metrics
    
    # carefull per patient split
    clf.fit(X_train_pat, y_train_pat)
    metrics = compute_metrics(clf, X_test_pat, y_test_pat)
    print(f"patient split : {metrics} \n")
    performance_dict[f'{clf_name}_pat'] = metrics
    
for clf_name, clf in classifiers.items():
    # Train on 2 patients test on the other
    auc_dict = auc_patient_cv(clf, X_pat, y_pat)
    print(f'train on 2 out of 3 : {auc_dict}')
    

#%% Simple LGBM

# create and fit LGMB
import lightgbm
lgbm_clf = lightgbm.LGBMClassifier(n_estimators = 500, objective = 'binary', class_weight = 'balanced')
lgbm_clf.fit(X_train_pat, y_train_pat, eval_metric = 'auc', verbose = 2)


# Compute metrics and plot ROC
metrics_dict = compute_metrics(lgbm_clf, X_test_pat, y_test_pat)
print(f'extratrees train on 2/3rd of each patient : {metrics_dict}')

y_pred_proba = lgbm_clf.predict_proba(X_test_pat)[:, [1]]
fpr, tpr, thresholds = roc_curve(y_test_pat, y_pred_proba, pos_label=1)

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - Extra Trees Classifier')
plt.plot(fpr, tpr)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#%% LGBM avoiding overfitting

lgbm_clf = lightgbm.LGBMClassifier(n_estimators = 500, objective = 'binary', 
                                   num_leaves = 100, min_data_in_leaf = 30, bagging_fraction = 0.8, 
                                   bagging_freq = 5, feature_fraction = 0.6, max_depth = 500,  
                                    lambda_l1 = 0.1) #is_unbalance = True,

lgbm_clf.fit(X_train_pat, y_train_pat, early_stopping_rounds=5, eval_set = [(X_test_pat, y_test_pat)], 
             eval_metric = 'auc', verbose = 2)

# Compute metrics and plot ROC
metrics_dict = compute_metrics(lgbm_clf, X_test_pat, y_test_pat)
print(f'extratrees train on 2/3rd of each patient : {metrics_dict}')

y_pred_proba = lgbm_clf.predict_proba(X_test_pat)[:, [1]]
fpr, tpr, thresholds = roc_curve(y_test_pat, y_pred_proba, pos_label=1)

plt.subplots(1, figsize=(10,10))
plt.title('Receiver Operating Characteristic - Extra Trees Classifier')
plt.plot(fpr, tpr)
plt.plot([0, 1], ls="--")
plt.plot([0, 0], [1, 0] , c=".7"), plt.plot([1, 1] , c=".7")
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#%% old basic clf
    
basic_clf = ExtraTreesClassifier()
basic_clf.fit(X_train_pat, y_train_pat)


# Evaluate performance before tuning
metrics_dict = compute_metrics(basic_clf, X_test_pat, y_test_pat)

y_pred_proba = basic_clf.predict_proba(X_test_pat)[:, [1]]
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
# from multiprocessing import Pool

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
#                   min_impurity_decrease = [0, 0.01, 0.25, 1, 10],
                     bootstrap = [False, True],
                     class_weight = ['balanced', 'balanced_subsample', None],
                     ccp_alpha = np.linspace(0, 0.08, 20) ,
                     max_samples = np.linspace(0.001, 0.5, 20).astype('float'))

# pool = Pool(4)

clf = EvolutionaryAlgorithmSearchCV(extra_trees_clf, 
                                    distributions, 
                                    population_size = 100,
                                    gene_mutation_prob = 0.10,
                                    gene_crossover_prob = 0.5,
                                    tournament_size = 3,
                                    generations_number = 6,
                                    cv = StratifiedKFold(n_splits = 5),
                                    verbose = 1,
                                    scoring = scorer_object,
                                    n_jobs = 1)
                                    #pmap = pool.map

search = clf.fit(X_train, y_train) # TODO test removing search

## best parameters found in search are {'criterion': 'entropy', 'n_estimators': 321,
# 'max_depth': 100, 'min_samples_split': 42, 'min_samples_leaf': 17, 'max_features': 'sqrt',
# 'max_leaf_nodes': 1000, 'min_impurity_decrease': 0, 'bootstrap': False, 
# 'class_weight': 'balanced_subsample', 'ccp_alpha': 0.004333333333333333, 
# 'max_samples': 0.10605263157894737}

#second run gave Best individual is: {'criterion': 'entropy', 'n_estimators': 956, 'max_depth': 150, 
# 'min_samples_split': 68, 'min_samples_leaf': 55, 'max_features': 'auto', 
# 'max_leaf_nodes': 1000, 'bootstrap': False, 'class_weight': 'balanced_subsample', 
# 'ccp_alpha': 0.05052631578947368, 'max_samples': 0.4737368421052632}
#   with fitness: 0.8395730353160524 ; {'auc': 0.8227768361581921, 'f1score': 0.5006485084306096, 'accuracy': 0.32019449161622426, 'balanced_accuracy': 0.540361581920904, 'recall': 0.772}
#%%

print(f'best parameters found in search are {clf.best_params_}')
#%% Evaluate best extra trees

#tuned_extra_trees = pickle.load( open( "neurovista_model_allpat_2feb_hyperparam_genetic_long_v2.sav", "rb" ) )
tuned_extra_trees.fit(X_train_pat, y_train_pat)
metrics_dict = compute_metrics(tuned_extra_trees, X_test_pat, y_test_pat)
print(f'extratrees trained on 2/3rd of each patient : {metrics_dict}')


#%% save and evaluate

#best_clf = ExtraTreesClassifier(n_estimators = 837, max_depth = 38, min_samples_leaf = 1, min_impurity_decrease = 0, criterion = 'entropy', bootstrap = False, min_samples_split = 2, class_weight = 'balanced')

#best_clf = ExtraTreesClassifier(**clf.best_params_)
#best_clf = best_clf.fit(X_train, y_train)

best_clf = clf.best_estimator_
metrics_dict = compute_metrics(best_clf, X_test, y_test)
print(f'model trained on all patients has following performance metrics after hyperparam tuning: {metrics_dict}')

import pickle
filename = 'neurovista_model_allpat_2feb_hyperparam_genetic_long_v2.sav'
pickle.dump(best_clf, open(filename, 'wb'))

#%% evaluate

from sklearn.model_selection import cross_val_score, KFold

paper_model = ExtraTreesClassifier(n_estimators=5000, random_state=0, max_depth=15, n_jobs=2,criterion='entropy', min_samples_split=7)
# gets 0.84 avg auc with 10fold, lowest fold 0.61 then 0.78
cv_auc = cross_val_score(paper_model, X, np.ravel(y), scoring = scorer_object, cv = KFold(10))
print(cv_auc.sum()/10)


#%%
#with open(r"neurovista_model_pat3_31jan_hyperparam_genetic.sav", "rb") as input_file:
#    best_clf = pickle.load(input_file)

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