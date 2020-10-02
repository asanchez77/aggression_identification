#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  7 17:23:19 2020

@author: 

categories : toxic, severe_toxic, obscene, threat, insult, identity_hate 

"""


#%%

"""Load the data """

import os
import pandas as pd
import numpy as np

DATA_PATH = "data/"

mode = "test"
focus_label = "severe_toxic"


def load_aggression_data_file (csvfile, housing_path = DATA_PATH):
    csv_path = os.path.join(housing_path, csvfile)
    return pd.read_csv(csv_path,header=0)

def load_training_data(csvfile):
    agg_data = load_aggression_data_file(csvfile)
    """Drop the information not used: facebook identifier"""
    agg_data = agg_data.drop('id', axis=1)    
    #Rename the columns
    agg_data = agg_data.rename(columns={'comment_text':"comment",
                                        focus_label:"toxic_label"})
    print(agg_data["comment"])
    print(agg_data["toxic_label"])
    # Obtain the labels and the comments
    agg_labels  = np.array(agg_data["toxic_label"]).reshape(-1,1)
    agg_comments = agg_data["comment"]
    return [agg_labels, agg_comments]

def load_testing_data(csvfile,csvfile_labels,label):
    agg_data = load_aggression_data_file(csvfile)
    agg_labels_temp = load_aggression_data_file(csvfile_labels)
    agg_labels= agg_labels_temp.replace(to_replace = -1, value = 0)
    """Drop the information not used: facebook identifier"""
    #agg_data = agg_data.drop('id', axis=1)    
    #Rename the columns
    agg_data = agg_data.rename(columns={'comment_text':"comment"})
    #agg_data["toxic_label"]=  agg_labels.replace(to_replace = -1, value = 0)
    print(agg_data["comment"])
    #print(agg_data["toxic_label"])
    # Obtain the labels and the comments
    agg_labels  = np.array(agg_labels[label]).reshape(-1,1)
    agg_comments = agg_data["comment"]
    return [agg_labels, agg_comments]

[agg_labels_train, agg_comments_train] = load_training_data("train.csv")
[agg_labels_dev, agg_comments_dev] = load_testing_data("test.csv","test_labels.csv",focus_label)

agg_labels_original = agg_labels_train.copy()
agg_labels_dev_original = agg_labels_dev.copy()

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

if focus_label=='toxic':
    clf_log_reg_filename = 'toxic_toxic_clf.sav'
if focus_label=='severe_toxic':
    clf_log_reg_filename = 'toxic_severe_toxic_clf.sav'
if focus_label=='obscene':
    clf_log_reg_filename = 'toxic_obscene_clf.sav'
if focus_label=='threat':
    clf_log_reg_filename = 'toxic_threat_clf.sav'
if focus_label=='insult':
    clf_log_reg_filename = 'toxic_insult_clf.sav'
if focus_label=='identity_hate':
    clf_log_reg_filename = 'toxic_identity_hate_clf.sav'

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

clf_toxic = Pipeline([('tfidf', TfidfVectorizer(vocabulary=vocab) ),
              ('clf', DecisionTreeClassifier(random_state=1234,
                                         ))
                ])

clf_severe_toxic = Pipeline([('tfidf', TfidfVectorizer(vocabulary=vocab) ),
              ('clf', DecisionTreeClassifier(random_state=1234,
                                         ))
                ])

clf_obscene = Pipeline([('tfidf', TfidfVectorizer(vocabulary=vocab) ),
              ('clf', DecisionTreeClassifier(random_state=1234,
                                         ))
                ])

clf_threat = Pipeline([('tfidf', TfidfVectorizer(vocabulary=vocab) ),
              ('clf', DecisionTreeClassifier(random_state=1234,
                                         ))
                ])

clf_insult = Pipeline([('tfidf', TfidfVectorizer(vocabulary=vocab) ),
              ('clf', DecisionTreeClassifier(random_state=1234,
                                         ))
                ])

clf_identity_hate = Pipeline([('tfidf', TfidfVectorizer(vocabulary=vocab) ),
              ('clf', DecisionTreeClassifier(random_state=1234,
                                         ))
                ])

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    

    if focus_label=='toxic':
        clf_current = clf_toxic
        clf_filename = 'toxic_toxic_clf_DT.sav'
        img_filename = 'toxic_toxic_img_DT.pdf'
        csv_sample_filename = 'toxic_toxic_sample_comments_DT.csv'
        txt_filename =  'toxic_toxic_tree.txt'
    if focus_label=='severe_toxic':
        clf_current = clf_severe_toxic
        clf_filename = 'toxic_severe_toxic_clf_DT.sav'
        img_filename = 'toxic_severe_img_DT.pdf'
        csv_sample_filename = 'toxic_severe_toxic_sample_comments_DT.csv'
        txt_filename =  'toxic_severe_toxic_tree.txt'
    if focus_label=='obscene':
        clf_current = clf_obscene
        clf_filename = 'toxic_obscene_clf_DT.sav'
        img_filename = 'toxic_obscene_img_DT.pdf'
        csv_sample_filename = 'toxic_obscene_sample_comments_DT.csv'
        txt_filename =  'toxic_obscene_tree.txt'
    if focus_label=='threat':
        clf_current = clf_threat
        clf_filename = 'toxic_threat_clf_DT.sav'
        img_filename = 'toxic_threat_img_DT.pdf'
        csv_sample_filename = 'toxic_threat_sample_comments_DT.csv'
        txt_filename =  'toxic_threat_tree.txt'
    if focus_label=='insult':
        clf_current = clf_insult
        clf_filename = 'toxic_insult_clf_DT.sav'
        img_filename = 'toxic_insult_img_DT.pdf'
        csv_sample_filename = 'toxic_insult_sample_comments_DT.csv'
        txt_filename =  'toxic_insult_tree.txt'
    if focus_label=='identity_hate':
        clf_current = clf_identity_hate
        clf_filename = 'toxic_identity_hate_clf_DT.sav'
        img_filename = 'toxic_identity_hate_img_DT.pdf'
        csv_sample_filename = 'toxic_identity_hate_sample_comments_DT.csv'
        txt_filename =  'toxic_identity_hate_tree.txt'
    

    print("Focus label:", focus_label)    
    print("pipeline:", [name for name, _ in clf_current.steps])
    print(clf_current['clf'])
    t0 = time()
    
    if mode == "train":
        print("Training model...")
        clf_current = clf_current.fit(agg_comments_train,agg_labels_train.ravel())
        print("Fit completed.")
        # save the model to disk
        joblib.dump(clf_current, clf_filename)
    else:
        print("Loading model...")
        clf_current = joblib.load(clf_filename)
        print("Finished loading model.")
    
    print("Predicting...")
    predicted = clf_current.predict(agg_comments_dev)
    print("Finished predicting.")
    #predicted = predicted.reshape(agg_labels_dev.shape)
    #print(predicted)
    
    print("F1 score: ", f1_score(agg_labels_dev, predicted, average='macro'))
    print("Precision score: ", precision_score(agg_labels_dev, predicted, average='macro'))
    print("Recall score: ", recall_score(agg_labels_dev, predicted, average='macro'))

    #print("comparing")
    #for real_label, predicted_label in zip(agg_labels_dev_encoded, predicted):
        #print(real_label, predicted_label)
      
#%%
        
features_ = clf_current[0].get_feature_names()
text_representation = tree.export_text(clf_current[1],feature_names = features_)
print(text_representation)
with open(txt_filename, 'w') as f:
    f.write(text_representation)

#%%

fig = plt.figure(figsize=(80, 30))
tree.plot_tree(clf_current[1], 
                   feature_names=features_,  
                   class_names=["OTHER", "OAG"],
                   filled=True,
                   #max_depth = 5,
                   fontsize=14)
fig.savefig(img_filename)