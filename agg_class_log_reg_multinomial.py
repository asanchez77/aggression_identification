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

DATA_PATH = "data/"

def load_aggression_data (csvfile, housing_path = DATA_PATH):
    csv_path = os.path.join(housing_path, csvfile)
    return pd.read_csv(csv_path,header=None)

agg_data_train = load_aggression_data("agr_en_train.csv")
agg_data_dev = load_aggression_data("agr_en_dev.csv")

#%%
"""Drop the information not used: facebook identifier"""
agg_data_train = agg_data_train.drop(0, axis=1)
agg_data_dev = agg_data_dev.drop(0, axis=1)

#%%

#Rename the columns
agg_data_train = agg_data_train.rename(columns={1:"comment",2:"agg_label"})
agg_data_dev = agg_data_dev.rename(columns={1:"comment",2:"agg_label"})


#%%
#print(agg_data.head())
print(agg_data_train["comment"])
print(agg_data_train["agg_label"])

print(agg_data_dev["comment"])
print(agg_data_dev["agg_label"])

#%%
#agg_data_train.info()

#%%
print(agg_data_train.describe())
print(agg_data_dev.describe())

#%%
# Obtain the labels and the comments

agg_labels_train  = np.array(agg_data_train["agg_label"]).reshape(-1,1)
agg_comments_train = agg_data_train["comment"]

agg_labels_dev  = np.array(agg_data_dev["agg_label"]).reshape(-1,1)
agg_comments_dev = agg_data_dev["comment"]

#%%

from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder_train = OrdinalEncoder()

agg_labels_train_encoded = ordinal_encoder_train.fit_transform(agg_labels_train)

#%%
print(agg_labels_train_encoded[:10])
print(ordinal_encoder_train.categories_)

ordinal_encoder_dev = OrdinalEncoder()

agg_labels_dev_encoded = ordinal_encoder_dev.fit_transform(agg_labels_dev)

#%%
print(agg_labels_dev_encoded[:10])
print(ordinal_encoder_dev.categories_)

#%%

from pprint import pprint
from time import time
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score

#%%

parameters = {
        'clf__penalty': ('l2',),
        'clf__multi_class': ('multinomial',),
        'clf__solver': ('lbfgs',),
        'clf__C': (5.0,),
        }
    

#%%

clf = Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', 
                                        ngram_range=(1, 5), lowercase=True) ),
              ('clf', LogisticRegression(penalty = 'l2',
                                         multi_class = 'multinomial' ,
                                         solver='lbfgs',
                                         C= 5.0,
                                         max_iter = 300))
                ])
#%%

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    print("Training model...")
    #print("Performing grid search...")
    print("pipeline:", [name for name, _ in clf.steps])
    print("parameters:")
    #pprint(parameters)
    t0 = time()
    clf = clf.fit(agg_comments_train,agg_labels_train_encoded.ravel())
    print("Fit completed.")
    predicted = clf.predict(agg_comments_dev)

    predicted = predicted.reshape(agg_labels_dev_encoded.shape)
    print(predicted)
    
    print("F1 score: ", f1_score(agg_labels_dev_encoded, predicted, average='macro'))
    
    #print("comparing")
    #for real_label, predicted_label in zip(agg_labels_dev_encoded, predicted):
        #print(real_label, predicted_label)
      
    

