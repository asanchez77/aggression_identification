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

DATA_PATH = "data/eng/"

mode = "train"
focus_label = 'OAG'


def load_aggression_data_file (csvfile, housing_path = DATA_PATH):
    csv_path = os.path.join(housing_path, csvfile)
    return pd.read_csv(csv_path,header=0)

def load_aggresion_data(csvfile):
    agg_data = load_aggression_data_file(csvfile)
    """Drop the information not used: facebook identifier"""
    agg_data = agg_data.drop('ID', axis=1)    
    #Rename the columns
    """*************************  IMPORTANT  *************************"""
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
agg_labels_original = agg_labels_train.copy()

def redifine_labels(agg_labels, focus_label):
    for i in range(len(agg_labels)):
        if agg_labels[i] != focus_label:
            agg_labels[i] = "ANOTHER"
    print (agg_labels)
    return agg_labels


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
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score

from sklearn.tree import DecisionTreeClassifier 
from sklearn import tree
from matplotlib import pyplot as plt


# to save model import joblib
import joblib 
#%%

"""Import the logistic regression model to obtain the top neg and top pos 
features: """

if focus_label=='NAG':
    clf_log_reg_filename = 'trac2_NAG_clf.sav'
if focus_label=='CAG':
    clf_log_reg_filename = 'trac2_CAG_clf.sav'
if focus_label=='OAG':
    clf_log_reg_filename = 'trac2_OAG_clf.sav'
if focus_label=='GEN':
    clf_log_reg_filename = 'trac2_GEN_clf.sav'

print("Loading model...")
clf_log_reg = joblib.load(clf_log_reg_filename)
print(clf_log_reg_filename," model loaded.")

#%%

from scipy.sparse import csr_matrix
coefs = clf_log_reg.named_steps["clf"].coef_

#%%
if type(coefs) == csr_matrix:
    coefs.toarray().tolist()[0]
else:
    coefs.tolist()
    
feature_names = clf_log_reg.named_steps["tfidf"].get_feature_names()


coefs_and_features = list(zip(coefs[0], feature_names))# Most positive features

neg_features = sorted(coefs_and_features, key=lambda x: x[0])# Most negative features

predictive_features = sorted(coefs_and_features, 
                             key=lambda x: x[0], 
                             reverse=True)# Most predictive overall
n_display_values = 30

most_neg = neg_features[:n_display_values]
most_pred = predictive_features[:n_display_values]

select_feats =   most_pred + most_neg

#%%

vocab = {x[1]: i for i, x in enumerate(select_feats)}


#%%

clf_NAG = Pipeline([('tfidf', TfidfVectorizer(vocabulary=vocab) ),
              ('clf', DecisionTreeClassifier(random_state=1234,
                                         ))
                ])

clf_CAG = Pipeline([('tfidf', TfidfVectorizer(vocabulary=vocab) ),
              ('clf', DecisionTreeClassifier(random_state=1234,
                                         ))
                ])

clf_OAG = Pipeline([('tfidf', TfidfVectorizer(vocabulary=vocab) ),
              ('clf', DecisionTreeClassifier(random_state=1234,
                                         ))
                ])

clf_GEN = Pipeline([('tfidf', TfidfVectorizer(vocabulary=vocab) ),
              ('clf', DecisionTreeClassifier(random_state=1234,
                                         ))
                ])

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    
    if focus_label=='NAG':
        clf_current = clf_NAG
        clf_filename = 'trac2_NAG_clf_DT.sav'
        img_filename = 'trac2_NAG_img_DT.pdf'
        csv_sample_filename = 'trac2_NAG_sample_comments_DT.csv'
    if focus_label=='CAG':
        clf_current = clf_CAG
        clf_filename = 'trac2_CAG_clf_DT.sav'
        img_filename = 'trac2_CAG_img_DT.pdf'
        csv_sample_filename = 'trac2_CAG_sample_comments_DT.csv'
    if focus_label=='OAG':
        clf_current = clf_OAG
        clf_filename = 'trac2_OAG_clf_DT.sav'
        img_filename = 'trac2_OAG_img_DT.pdf'
        csv_sample_filename = 'trac2_OAG_sample_comments_DT.csv'
    if focus_label=='GEN' or focus_label=='NGEN':
        clf_current = clf_GEN
        clf_filename = 'trac2_GEN_clf_DT.sav'
        img_filename = 'trac2_GEN_img_DT.pdf'
        csv_sample_filename = 'trac2_GEN_sample_comments_DT.csv'

    print("Focus label:", focus_label)
    print("pipeline:", [name for name, _ in clf_current.steps])
    print(clf_current['clf'])
    t0 = time()
    
    if mode == "train":
        print("Training model...")
        clf_current = clf_current.fit(agg_comments_train,agg_labels_train_encoded.ravel())
        print("Fit completed.")
        # save the model to disk
        joblib.dump(clf_current, clf_filename)
    else:
        print("Loading model")
        clf_current = joblib.load(clf_filename)
    
    predicted = clf_current.predict(agg_comments_dev)

    predicted = predicted.reshape(agg_labels_dev_encoded.shape)
    print(predicted)
    
    print("F1 score: ", f1_score(agg_labels_dev_encoded, predicted, average='macro'))
    print("Precision score: ", precision_score(agg_labels_dev_encoded, predicted, average='macro'))
    print("Recall score: ", recall_score(agg_labels_dev_encoded, predicted, average='macro'))
    #print("comparing")
    #for real_label, predicted_label in zip(agg_labels_dev_encoded, predicted):
        #print(real_label, predicted_label)
      
#%%

text_representation = tree.export_text(clf_current[1])
print(text_representation)
#%%

features_ = clf_current[0].get_feature_names()

fig = plt.figure(figsize=(80, 30))
tree.plot_tree(clf_current[1], 
                   feature_names=features_,  
                   class_names=["OTHER", "OAG"],
                   filled=True,
                   #max_depth = 5,
                   fontsize=14)
fig.savefig(img_filename)