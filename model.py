# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 02:04:40 2021

@author: gijsb
"""
#X_train_save = X_train
#y_train_save = y_train
#TODO remove the first line of zeroes from label and X_train

X_train = np.nan_to_num(X_train)

#%% find labels from filenames

X_train_bis

train_data_path = 'C:/Users/gijsb/OneDrive/Documents/epilepsy_neurovista_data/TrainPat1'
filelist = [join(train_data_path, f) for f in listdir(train_data_path) if isfile(join(train_data_path, f))]


label = [0]
for file_name_long in filelist:
    label.append(file_name_long[-5: -4])

label_array = np.array(label).astype('int')
label_array = np.reshape(label_array, (len(label),1))


#%% Test train split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_train_save, y_train_save, test_size=0.33, random_state=42)


#fit model
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=100, random_state=0)
clf.fit(X_train, y_train)

# Evaluate model


from sklearn.metrics import roc_auc_score
score = roc_auc_score(y_test, clf.predict_proba(X_test)[:, 1])

def scoring_auc(clf, X, y):
    score = roc_auc_score(y, clf.predict_proba(X)[:, 1])
    return score


print(f'AUC : {score}')


#%% save model to pickle

import pickle
filename = 'neurovista_pat2_28jan.sav'
pickle.dump(clf, open(filename, 'wb'))

#%% Randomised search on hjyperparameters

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np

#def scoring_auc(y_true, y_pred):
#    score = roc_auc_score(y_true, y_pred)
#    return score

extra_trees_clf = ExtraTreesClassifier(n_estimators=100, criterion='gini', n_jobs=4)

distributions = dict(n_estimators=np.linspace(10, 500, num=490).astype('int'), 
                     max_depth=np.linspace(4, 40, num=30), min_samples_split=[1,2,3],  
                     min_samples_leaf=[1,2], 
                     min_impurity_decrease=[0, 0.1], 
                     bootstrap=[False, True])

clf = RandomizedSearchCV(extra_trees_clf, distributions, scoring = scoring_auc, random_state=0)

search = clf.fit(X_train, label_array)

#%%

print(search.cv_results_)
cv_clf = search.best_estimator_
#TODO add fil number to submisison ; check why some 0 and 1 have the same number
