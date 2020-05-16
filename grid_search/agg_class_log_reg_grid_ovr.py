#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 17:23:19 2020

@author: 
"""


#%%

"""Load the data """

import os
import pandas as pd
import numpy as np

DATA_PATH = "/home/armando/ml/tf/py_code/agg/"

def load_aggression_data (housing_path = DATA_PATH):
    csv_path = os.path.join(housing_path, "agr_en_train.csv")
    return pd.read_csv(csv_path,header=None)

agg_data = load_aggression_data()

#%%
"""Drop the information not used: facebook identifier"""
agg_data = agg_data.drop(0, axis=1)

#%%

#Rename the columns
agg_data = agg_data.rename(columns={1:"comment",2:"agg_label"})


#%%
#print(agg_data.head())
print(agg_data["comment"])
print(agg_data["agg_label"])

#%%
agg_data.info()

#%%
print(agg_data.describe())

#%%
# Obtain the labels and the comments

agg_labels  = np.array(agg_data["agg_label"]).reshape(-1,1)
agg_comments = agg_data["comment"]

#%%

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()

agg_labels_encoded = ordinal_encoder.fit_transform(agg_labels)

#%%
print(agg_labels_encoded[:10])
print(ordinal_encoder.categories_)


#%%

from pprint import pprint
from time import time
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

#%%

pipelines = [
    Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', 
                                        ngram_range=(1, 5), lowercase=True) ),
              ('clf', LogisticRegression() )
                ]),
    Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', 
                                        ngram_range=(1, 5), lowercase=True) ),
              ('clf', LogisticRegression() )
                ]),
    Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', 
                                        ngram_range=(1, 5), lowercase=True) ),
              ('clf', LogisticRegression() )
                ]),
            ]

#%%

parameters = [
    {
        #'clf__max_iter': (90,),
        'clf__penalty': ('l1','l2',),
        'clf__multi_class': ('ovr',),
        'clf__solver': ('liblinear',),
        'clf__C': (0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 300.0, 500.0),
        #'clf__C': (6.0, 8.0, 10.0, 20.0, 30.0, 40.0),
        'clf__max_iter': (200,),
        },
    {#'clf__max_iter': (90,),
        'clf__penalty': ('l2',),
        'clf__multi_class': ('ovr',),
        'clf__solver': ('lbfgs','newton-cg','sag',),
        'clf__C': (0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 300.0, 500.0),
        #'clf__C': (6.0, 8.0, 10.0, 20.0, 30.0, 40.0),
        'clf__max_iter': (200,),
        },
    {#'clf__max_iter': (90,),
        'clf__penalty': ('elasticnet',),
        'clf__l1_ratio':(0, 0.5, 1,),
        'clf__multi_class': ('ovr',),
        'clf__solver': ('saga',),
        'clf__C': (0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0, 200.0, 300.0, 500.0),
        #'clf__C': (6.0, 8.0, 10.0, 20.0, 30.0, 40.0),
        'clf__max_iter': (200,),
        }
    ]
#%%

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for classifier
    print("Performing grid search...")
    
    for text_clf, param in zip(pipelines, parameters):
        gs_clf = GridSearchCV(text_clf, param, n_jobs=-1, scoring='f1_macro', verbose=2)
        print("Performing grid search...")
        print("pipeline:", [name for name, _ in text_clf.steps])
        print("parameters:")
        pprint(param)
        t0 = time()
        gs_clf = gs_clf.fit(agg_comments,agg_labels_encoded.ravel())
        print("done in %0.3fs" % (time() - t0))
        print()
        print("Best score: %0.3f" % gs_clf.best_score_)
        print("Best parameters set:")
        best_parameters = gs_clf.best_estimator_.get_params()
        for param_name in sorted(param.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
      
    

