#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 16:42:17 2020

@author:
"""
#%%

"""The classes that will be included in the histogram"""
eval_classes = ['CAG']
"""The total number of iterations"""
iter_val = 2
#%%

"""Load the data """

import os
import pandas as pd
import numpy as np

DATA_PATH = "data/"



def load_aggression_data_file (csvfile, housing_path = DATA_PATH):
    csv_path = os.path.join(housing_path, csvfile)
    return pd.read_csv(csv_path,header=None)

def load_aggresion_data(csvfile):
    agg_data = load_aggression_data_file(csvfile)
    """Drop the information not used: facebook identifier"""
    agg_data = agg_data.drop(0, axis=1)    
    #Rename the columns
    agg_data = agg_data.rename(columns={1:"comment",2:"agg_label"})
    print(agg_data["comment"])
    print(agg_data["agg_label"])
    # Obtain the labels and the comments
    agg_labels  = np.array(agg_data["agg_label"]).reshape(-1,1)
    agg_comments = agg_data["comment"]
    return [agg_labels, agg_comments]

[agg_labels_train, agg_comments_train] = load_aggresion_data("agr_en_train.csv")
[agg_labels_dev, agg_comments_dev] = load_aggresion_data("agr_en_dev.csv")

def redifine_labels(agg_labels, focus_label):
    for i in range(len(agg_labels)):
        if agg_labels[i] != focus_label:
            agg_labels[i] = "ANOTHER"
    print (agg_labels)
    return agg_labels

focus_label = 'NAG'
agg_labels_train = redifine_labels(agg_labels_train, focus_label)
agg_labels_dev = redifine_labels(agg_labels_dev, focus_label)

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

#%%

clf_NAG = Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', 
                                        ngram_range=(1, 5), lowercase=True) ),
              ('clf', LogisticRegression(penalty = 'l2',
                                         multi_class = 'ovr' ,
                                         solver='liblinear',
                                         C= 5.0,
                                         #max_iter = 300))
                                         ))
                ])

clf_CAG = Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', 
                                        ngram_range=(1, 5), lowercase=True) ),
              ('clf', LogisticRegression(penalty = 'l2',
                                         multi_class = 'ovr' ,
                                         solver='liblinear',
                                         C= 100.0,
                                         #max_iter = 300))
                                         ))
                ])

clf_OAG = Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', 
                                        ngram_range=(1, 5), lowercase=True) ),
              ('clf', LogisticRegression(penalty = 'l1',
                                         multi_class = 'ovr' ,
                                         solver='liblinear',
                                         C= 10.0,
                                         #max_iter = 300))
                                         ))
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

    total_false_negatives = pd.DataFrame()
    
    total_false_positives = pd.DataFrame()
    
    for i in range(0,iter_val):
        print("Iteration number ",i)
        false_negative_dataset = agg_comments_train
        #false_negative_dataset = agg_comments_dev
        false_negative_labels_encoded = agg_labels_train_encoded
        #false_negative_labels_encoded = agg_labels_dev_encoded
    
        print("pipeline:", [name for name, _ in clf_current.steps])
        print(clf_current['clf'])
        t0 = time()
        clf_current = clf_current.fit(agg_comments_train,agg_labels_train_encoded.ravel())
        print("Fit completed.")
        
        
        predicted = clf_current.predict(false_negative_dataset)
    
        predicted = predicted.reshape(false_negative_labels_encoded.shape)
        print(predicted)
        
        print("F1 score: ", f1_score(false_negative_labels_encoded, predicted, average='macro'))
        
        """Obtain false negatives"""
        [false_negatives, false_negatives_index] = obtain_false_negatives(
            predicted,
            false_negative_labels_encoded,
            false_negative_dataset)
        total_false_negatives = pd.concat([total_false_negatives,
                                          pd.DataFrame(false_negatives_index)],
                                         ignore_index = True, 
                                         axis =1)
        """Obtain false positives"""
        [false_positives, false_positives_index] = obtain_false_positives(
            predicted,
            false_negative_labels_encoded,
            false_negative_dataset)
        total_false_positives = pd.concat([total_false_positives,
                                          pd.DataFrame(false_positives_index)],
                                         ignore_index = True, 
                                         axis =1)
        
        
        
        clf_current['clf'].random_state = random.randint(1,1000)
        #print("comparing")
        #for real_label, predicted_label in zip(agg_labels_dev_encoded, predicted):
            #print(real_label, predicted_label)

      
#%%

print(total_false_negatives)
print(total_false_positives)

#%%
"""Write to CSV file"""
print("*********************")
#print(false_negatives)

false_negatives_df =  pd.DataFrame(list(false_negatives))
false_negatives_df = false_negatives_df.rename(columns={0:focus_label+"_false_negative"})

print("Creating false negatives file")
joined_df = pd.concat([false_negatives_df], axis=1, sort=False)
joined_df.to_csv('trac2_false_negatives.csv')