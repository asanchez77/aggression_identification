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
iter_val = 100

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
focus_label = 'NAG'
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
    
def obtain_false_negatives(predicted,labels_encoded,comments):
    false_negatives = []
    false_negatives_index = []
    for i in range(len(predicted)):
        if predicted[i]!=labels_encoded[i] and predicted[i]==0:
            #print("-------")
            #print(false_negative_dataset[i])
            false_negatives_index.append(i)
            false_negatives.append(comments[i])
    return [false_negatives, false_negatives_index]

#%%
    
def obtain_false_positives(predicted,labels_encoded,comments):
    false_positives = []
    false_positives_index = []
    for i in range(len(predicted)):
        if predicted[i]!=labels_encoded[i] and predicted[i]==1:
            #print("-------")
            #print(false_negative_dataset[i])
            false_positives_index.append(i)
            false_positives.append(comments[i])
    return [false_positives, false_positives_index]

#%%


from time import time
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import random

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
    total_false_negatives = pd.DataFrame()
    
    false_negative_dataset = agg_comments_train
    #false_negative_dataset = agg_comments_dev
    false_negative_labels_encoded = agg_labels_train_encoded
    #false_negative_labels_encoded = agg_labels_dev_encoded

    print("pipeline:", [name for name, _ in clf_current.steps])
    print(clf_current['clf'])
    t0 = time()
    clf_current = clf_current.fit(agg_comments_train,agg_labels_train_encoded.ravel())
    print("Fit completed.")
    
    
    predicted_prob = clf_current.predict_proba(false_negative_dataset)
#%%
    #predicted = predicted.reshape(false_negative_labels_encoded.shape)
    #print(predicted)
#%%

#print("F1 score: ", f1_score(false_negative_labels_encoded, predicted, average='macro'))
for i in range(0,iter_val):
    predicted = [];
    print("Iteration number ",i)
    for j in range(len(predicted_prob)):
        predicted.append(np.random.choice([0,1],1,p=predicted_prob[j].tolist()))
    
    [false_negatives, false_negatives_index] = obtain_false_negatives(
        predicted,
        false_negative_labels_encoded,
        false_negative_dataset)
    total_false_negatives = pd.concat([total_false_negatives,
                                      pd.DataFrame(false_negatives_index)],
                                     ignore_index = True, 
                                     axis =1)

    
#%%

print(total_false_negatives)

np_total_false_neg = total_false_negatives.to_numpy()
unique_values = np.unique(np_total_false_neg)
freq =[]
total_freq = 0
for k  in range(len(unique_values)):
    freq.append(np.count_nonzero(np_total_false_neg ==unique_values[k]))
    total_freq = total_freq + freq[k]

relative_freq = []
for k  in range(len(unique_values)):
    relative_freq.append(freq[k] / total_freq)

np_freq = np.asarray(freq)
np_freq = np_freq.reshape(-1,1)
np_relative_freq = np.asarray(relative_freq)
np_relative_freq = np_relative_freq.reshape(-1,1)

unique_values = unique_values.reshape(-1,1)

con = np.concatenate((unique_values,np_relative_freq),axis=1)
pd_con = pd.DataFrame(con)
print(pd_con)
pd_sorted = pd_con.sort_values(by= 1,ascending=False)
print(pd_sorted)
np_total_false_neg = np_total_false_neg.reshape(-1)

#%%

from matplotlib import pyplot

fig, ax = pyplot.subplots()
pyplot.title(focus_label)
#ax = fig.add_axes([0,0,1,1])
#ax.bar(unique_values, freq)
ax.bar(range(len(pd_sorted[0])),pd_sorted[1].to_numpy())
threshold = 0.9*pd_sorted[1].to_numpy()[0]
ax.plot([0., len(pd_sorted[0])], [threshold, threshold], "k--")
pyplot.xticks(rotation=90, ha='center')
pyplot.show()


#%%
#Write to CSV file
print("*********************")
#print(false_negatives)

pd_sorted = pd_sorted.rename(columns={0:focus_label+"_false_negative_idx", 1:"relative frequency"})

print("Creating false negatives file")
joined_df = pd.concat([pd_sorted], axis=1, sort=False)
joined_df.to_csv('trac2_false_negatives.csv')

