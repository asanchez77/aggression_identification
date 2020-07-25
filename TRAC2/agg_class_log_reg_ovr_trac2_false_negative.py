#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 17:23:19 2020

@author: 
"""
#%%

"""The classes that will be included in the histogram"""
eval_classes = ['CAG']
"""The total number of iterations"""
iter_val = 3

#%%

"""Load the data """

import os
import pandas as pd
import numpy as np

DATA_PATH = "data/eng/"


def load_aggression_data_file (csvfile, housing_path = DATA_PATH):
    csv_path = os.path.join(housing_path, csvfile)
    return pd.read_csv(csv_path,header=0)

def load_aggresion_data(csvfile):
    agg_data = load_aggression_data_file(csvfile)
    """Drop the information not used: facebook identifier"""
    agg_data = agg_data.drop('ID', axis=1)    
    #Rename the columns
    """For *AG use Sub-task A and for *GEN use Sub-task B to obtain the 
    labels used for training"""
    agg_data = agg_data.rename(columns={'Text':"comment",'Sub-task A':"agg_label"})
    print(agg_data["comment"])
    print(agg_data["agg_label"])
    # Obtain the labels and the comments
    agg_labels  = np.array(agg_data["agg_label"]).reshape(-1,1)
    agg_comments = agg_data["comment"]
    return [agg_labels, agg_comments]

[agg_labels_train, agg_comments_train] = load_aggresion_data("trac2_eng_train.csv")
[agg_labels_dev, agg_comments_dev] = load_aggresion_data("trac2_eng_dev.csv")

def redifine_labels(agg_labels, focus_label):
    for i in range(len(agg_labels)):
        if agg_labels[i] != focus_label:
            agg_labels[i] = "ANOTHER"
    print (agg_labels)
    return agg_labels

"""OVR scheme """
focus_label = 'CAG'
agg_labels_train = redifine_labels(agg_labels_train, focus_label)
agg_labels_dev = redifine_labels(agg_labels_dev, focus_label)

#%%
"""Enconde the training labels """
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder_train = OrdinalEncoder()
agg_labels_train_encoded = ordinal_encoder_train.fit_transform(agg_labels_train)

print(agg_labels_train_encoded[:10])
print(ordinal_encoder_train.categories_)
#%%
"""Encode the dev labels"""
ordinal_encoder_dev = OrdinalEncoder()
agg_labels_dev_encoded = ordinal_encoder_dev.fit_transform(agg_labels_dev)

print(agg_labels_dev_encoded[:10])
print(ordinal_encoder_dev.categories_)

#%%


from time import time
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

"""Create the pipelines with the best parameters for each class"""

clf_NAG = Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', 
                                        ngram_range=(1, 5), lowercase=True) ),
              ('clf', LogisticRegression(penalty = 'l1',
                                         multi_class = 'ovr' ,
                                         solver='liblinear',
                                         C= 10.0,
                                         max_iter = 200))
                                         #))
                ])

clf_CAG = Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', 
                                        ngram_range=(1, 5), lowercase=True) ),
              ('clf', LogisticRegression(penalty = 'l1',
                                         multi_class = 'ovr' ,
                                         solver='liblinear',
                                         C= 200.0,
                                         max_iter = 200))
                                         #))
                ])

clf_OAG = Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', 
                                        ngram_range=(1, 5), lowercase=True) ),
              ('clf', LogisticRegression(penalty = 'l1',
                                         multi_class = 'ovr' ,
                                         solver='liblinear',
                                         C= 50.0,
                                         max_iter = 200))
                                         #))
                ])

clf_GEN = Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', 
                                        ngram_range=(1, 5), lowercase=True) ),
              ('clf', LogisticRegression(penalty = 'l1',
                                         multi_class = 'ovr' ,
                                         solver='liblinear',
                                         C= 50.0,
                                         max_iter = 200))
                                         #))
                ])

#%%
"""Train the models and obtain false positives and false negatives"""
if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    print("Training model...")
    
    if focus_label=='NAG':
        clf_current = clf_NAG
    if focus_label=='CAG':
        clf_current = clf_CAG
    if focus_label=='OAG':
        clf_current = clf_OAG
    if focus_label=='GEN' or focus_label=='NGEN':
        clf_current = clf_GEN


    #false_negative_dataset = agg_comments_train
    false_negative_dataset = agg_comments_dev
    #false_negative_labels_encoded = agg_labels_train_encoded
    false_negative_labels_encoded = agg_labels_dev_encoded

    print("pipeline:", [name for name, _ in clf_current.steps])
    print(clf_current['clf'])
    t0 = time()
    clf_current = clf_current.fit(agg_comments_train,agg_labels_train_encoded.ravel())
    print("Fit completed.")
    predicted = clf_current.predict(false_negative_dataset)

    predicted = predicted.reshape(false_negative_labels_encoded.shape)
    print(predicted)
    
    print("F1 score: ", f1_score(false_negative_labels_encoded, predicted, average='macro'))
    #print("comparing")
    #for real_label, predicted_label in zip(agg_labels_dev_encoded, predicted):
        #print(real_label, predicted_label)

    false_negatives = []

    for i in range(len(predicted)):
        if predicted[i]!=false_negative_labels_encoded[i] and predicted[i]==0:
            #print("-------")
            #print(false_negative_dataset[i])
            false_negatives.append(false_negative_dataset[i])

    print("*********************")
    #print(false_negatives)

    false_negatives_df =  pd.DataFrame(list(false_negatives))
    false_negatives_df = false_negatives_df.rename(columns={0:focus_label+"_false_negative"})

    print("Creating false negatives file")
    joined_df = pd.concat([false_negatives_df], axis=1, sort=False)
    joined_df.to_csv('trac2_false_negatives.csv')

