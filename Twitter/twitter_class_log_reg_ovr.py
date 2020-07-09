#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 17:23:19 2020

@author: 
    
    
categories: 'abusive', 'hateful', 'normal', 'spam'
"""


#%%

"""Load the data """

import os
import pandas as pd
import numpy as np

DATA_PATH = "../../twitter_data/"



def load_aggression_data_file (csvfile, housing_path = DATA_PATH):
    csv_path = os.path.join(housing_path, csvfile)
    return pd.read_csv(csv_path,index_col = 0)

def load_aggresion_data(csvfile):
    agg_data = load_aggression_data_file(csvfile)
    """Drop the information not used: facebook identifier"""
    #agg_data = agg_data.drop('ID', axis=1)    
    #Rename the columns
    """For *AG use Sub-task A and for *GEN use Sub-task B to obtain the 
    labels used for training"""
    #agg_data = agg_data.rename(columns={0:"comment",1:"agg_label"})
    print(agg_data["comment"])
    print(agg_data["agg_label"])
    # Obtain the labels and the comments
    agg_labels  = np.array(agg_data["agg_label"]).reshape(-1,1)
    agg_comments = agg_data["comment"]
    return [agg_labels, agg_comments]

[agg_labels_train, agg_comments_train] = load_aggresion_data("hatespeech_text_train.csv")
[agg_labels_dev, agg_comments_dev] = load_aggresion_data("hatespeech_text_test.csv")

#%%

def redifine_labels(agg_labels, focus_label):
    for i in range(len(agg_labels)):
        if agg_labels[i] != focus_label:
            agg_labels[i] = "ANOTHER"
    print (agg_labels)
    return agg_labels

focus_label = 'hateful'
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
from time import time
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

#%%

clf_abusive = Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', 
                                        ngram_range=(2, 5), lowercase=True) ),
              ('clf', LogisticRegression(penalty = 'l1',
                                         multi_class = 'ovr' ,
                                         solver='liblinear',
                                         C= 1.0,
                                         max_iter = 200))
                                         #))
                ])

clf_hateful = Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', 
                                        ngram_range=(2, 5), lowercase=True) ),
              ('clf', LogisticRegression(penalty = 'l1',
                                         multi_class = 'ovr' ,
                                         solver='liblinear',
                                         C= 5.0,
                                         max_iter = 200))
                                         #))
                ])

clf_normal = Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', 
                                        ngram_range=(1, 5), lowercase=True) ),
              ('clf', LogisticRegression(penalty = 'l2',
                                         multi_class = 'ovr' ,
                                         solver='liblinear',
                                         C= 5.0,
                                         max_iter = 200))
                                         #))
                ])

clf_spam = Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', 
                                        ngram_range=(2, 5), lowercase=True) ),
              ('clf', LogisticRegression(penalty = 'l1',
                                         multi_class = 'ovr' ,
                                         solver='liblinear',
                                         C= 5.0,
                                         max_iter = 200))
                                         #))
                ])

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    print("Training model...")

    if focus_label=='abusive':
        clf_current = clf_abusive
    if focus_label=='hateful':
        clf_current = clf_hateful
    if focus_label=='normal':
        clf_current = clf_normal
    if focus_label == 'spam':
        clf_current = clf_spam

    print("pipeline:", [name for name, _ in clf_current.steps])
    print(clf_current['clf'])
    t0 = time()
    clf_current = clf_current.fit(agg_comments_train,agg_labels_train_encoded.ravel())
    print("Fit completed.")
    predicted = clf_current.predict(agg_comments_dev)

    predicted = predicted.reshape(agg_labels_dev_encoded.shape)
    print(predicted)
    
    print("Test F1 score: ", f1_score(agg_labels_dev_encoded, predicted, average='macro'))
    print("comparing")
    # for real_label, predicted_label in zip(agg_labels_dev_encoded, predicted):
    #     print(real_label, predicted_label)
      
#%%

from scipy.sparse import csr_matrix
coefs = clf_current.named_steps["clf"].coef_

#%%
if type(coefs) == csr_matrix:
    coefs.toarray().tolist()[0]
else:
    coefs.tolist()
    
#%%    
feature_names = clf_current.named_steps["tfidf"].get_feature_names()

#%%
coefs_and_features = list(zip(coefs[0], feature_names))# Most positive features
#%%
neg_features = sorted(coefs_and_features, key=lambda x: x[0])# Most negative features

#%%
predictive_features = sorted(coefs_and_features, 
                             key=lambda x: x[0],
                             reverse=True)# Most predictive overall

#%%


n_display_values = 30

most_neg = neg_features[:n_display_values]
most_pred = predictive_features[:n_display_values]


def print_format_coef(features_coef1, features_coef2):
    for feature1, feature2 in zip(features_coef1, features_coef2):
        repr_string1 = repr(feature1[1])
        repr_string1 = repr_string1[1:]
        repr_string1 = repr_string1[:-1]
        repr_string2 = repr(feature2[1])
        repr_string2 = repr_string2[1:]
        repr_string2 = repr_string2[:-1]
        print('\hline')
        print('\say{%s} &  %.2f &   \say{%s} &  %.2f \\\\ ' % (repr_string1,feature1[0], repr_string2, feature2[0]))
    return


#print("-------------")
print("features")
print_format_coef(most_neg,most_pred)
#print("-------------")
#print("most predictive features")
#print_format_coef(most_pred)

from matplotlib import pyplot

# get importance
importance = most_neg + most_pred[::-1]
#print(importance)

fig, ax = pyplot.subplots()
pyplot.title(focus_label)
ax.bar([repr(x[1])[1:-1] for x in importance], [x[0] for x in importance], -.9, 0,  align='edge')
pyplot.xticks(rotation=90, ha='right')
pyplot.show()


#%%
#sorted(coefs_and_features, key=lambda x: abs(x[0]), reverse=True)

#%%

