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

#%% Helper functions

def compute_metrics(clf, X_test, y_test):
    """
    return a dict containing auc, f1score, accuracy, balanced_accuracy, recall
    """
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)
    
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


#%% patient specific models

for p in [2]:
    X = np.load(f'neurovista_X_train_pat{p}.npy').astype('float32')
    X = np.nan_to_num(X)
    
    y = np.load(f'neurovista_y_train_pat{p}.npy').astype('float32')
    y = np.nan_to_num(y)
    

    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    
    
    #fit model
    
    clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
    clf.fit(X_train, y_train)
    
    # Evaluate model
      
    
    score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])
        
    print(f'model for patient {p} has AUC : {score}')

#%% Model for all patients with cross patient test/train split

X_pat = {}
y_pat = {}

# Load data from file
for p in [1,2,3]:
    X_pat[f'pat{p}'] = np.load(f'neurovista_X_train_pat{p}.npy').astype('float32')
    X_pat[f'pat{p}'] = np.nan_to_num(X_pat[f'pat{p}'])
    
    y_pat[f'pat{p}'] = np.load(f'neurovista_y_train_pat{p}.npy').astype('float32')
    y_pat[f'pat{p}'] = np.nan_to_num(y_pat[f'pat{p}'])

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

X_pat = {}
y_pat = {}

# Load data from file
for p in [1,2,3]:
    X_pat[f'pat{p}'] = np.load(f'neurovista_X_train_pat{p}.npy').astype('float32')
    X_pat[f'pat{p}'] = np.nan_to_num(X_pat[f'pat{p}'])
    
    y_pat[f'pat{p}'] = np.load(f'neurovista_y_train_pat{p}.npy').astype('float32')
    y_pat[f'pat{p}'] = np.nan_to_num(y_pat[f'pat{p}'])

# Assign data to train and test sets
X_train = np.vstack((X_pat['pat3'], X_pat['pat2']))
y_train = np.vstack((y_pat['pat3'], y_pat['pat2']))

#X_train = X_pat['pat3']
#y_train = y_pat['pat3']

X_test = np.nan_to_num(X_pat['pat1'].astype('float32'))
y_test = np.nan_to_num(y_pat['pat1'].astype('float32'))

del X_pat, y_pat

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