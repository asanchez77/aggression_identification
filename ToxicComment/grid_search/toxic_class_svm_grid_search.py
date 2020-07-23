#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu JUNE 25 17:23:19 2020

@author: 

Code used to find the best parameters for the SVM
classifier used for TOXIC COMMENTS
"""


#%%

"""Load the data """

import os
import pandas as pd
import numpy as np

DATA_PATH = "../data/"


def load_aggression_data_file (csvfile, housing_path = DATA_PATH):
    csv_path = os.path.join(housing_path, csvfile)
    return pd.read_csv(csv_path,header=0)

def load_aggresion_data(csvfile):
    agg_data = load_aggression_data_file(csvfile)
    """Drop the information not used: facebook identifier"""
    agg_data = agg_data.drop('id', axis=1)    
    #Rename the columns
    agg_data = agg_data.rename(columns={'comment_text':"comment",
                                        'threat':"toxic_label"})
    print(agg_data["comment"])
    print(agg_data["toxic_label"])
    # Obtain the labels and the comments
    agg_labels  = np.array(agg_data["toxic_label"]).reshape(-1,1)
    agg_comments = agg_data["comment"]
    return [agg_labels, agg_comments]

[agg_labels, agg_comments] = load_aggresion_data("train.csv")

#%%

from pprint import pprint
from time import time
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.svm import NuSVC

from sklearn.model_selection import GridSearchCV

#%%

pipelines = [
    Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', 
                                        ngram_range=(1, 5), lowercase=True) ),
              ('clf', NuSVC() )
                ]),
            ]

#%%


parameters = [
    {
        'clf__nu': (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        'clf__kernel': ('linear','poly',),
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
        gs_clf = gs_clf.fit(agg_comments,agg_labels.ravel())
        print("done in %0.3fs" % (time() - t0))
        print()
        print("Best score: %0.3f" % gs_clf.best_score_)
        print("Best parameters set:")
        best_parameters = gs_clf.best_estimator_.get_params()
        for param_name in sorted(param.keys()):
            print("\t%s: %r" % (param_name, best_parameters[param_name]))
    
      
    

