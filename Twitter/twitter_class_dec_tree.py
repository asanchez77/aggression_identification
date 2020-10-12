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

mode = "train"
focus_label = 'spam'
n_display_values = 30
max_depth_var = 5

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

agg_labels_original = agg_labels_train.copy()

#%%

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

if focus_label=='abusive':
    clf_log_reg_filename = 'twitter_abusive_clf.sav'
if focus_label=='hateful':
    clf_log_reg_filename = 'twitter_hateful_clf.sav'
if focus_label=='normal':
    clf_log_reg_filename = 'twitter_normal_clf.sav'
if focus_label=='spam':
    clf_log_reg_filename = 'twitter_spam_clf.sav'

print("Loading model...")
clf_log_reg = joblib.load(clf_log_reg_filename)
print(clf_log_reg_filename," model loaded.")
#%%
"""Now the most positive and most negative features are obtained from
each model and are used for the decision tree classifier"""


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


most_neg = neg_features[:n_display_values]
most_pred = predictive_features[:n_display_values]

select_feats =   most_pred + most_neg

#%%

vocab = {x[1]: i for i, x in enumerate(select_feats)}


#%%

clf_abusive = Pipeline([('tfidf', TfidfVectorizer(vocabulary=vocab) ),
              ('clf', DecisionTreeClassifier(random_state=1234,
                                             max_depth = max_depth_var,
                                         ))
                ])

clf_hateful = Pipeline([('tfidf', TfidfVectorizer(vocabulary=vocab) ),
              ('clf', DecisionTreeClassifier(random_state=1234,
                                             max_depth = max_depth_var,
                                         ))
                ])

clf_normal = Pipeline([('tfidf', TfidfVectorizer(vocabulary=vocab) ),
              ('clf', DecisionTreeClassifier(random_state=1234,
                                             max_depth = max_depth_var,
                                         ))
                ])

clf_spam = Pipeline([('tfidf', TfidfVectorizer(vocabulary=vocab) ),
              ('clf', DecisionTreeClassifier(random_state=1234,
                                             max_depth = max_depth_var,
                                         ))
                ])

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    if focus_label=='abusive':
        clf_current = clf_abusive
        clf_filename = 'twitter_abusive_clf_DT.sav'
        img_filename = 'twitter_abusive_img_DT.pdf'
        csv_sample_filename = 'twitter_abusive_sample_comments_DT.csv'
        txt_filename =  'twitter_abusive_tree.txt'
    if focus_label=='hateful':
        clf_current = clf_hateful
        clf_filename = 'twitter_hateful_clf_DT.sav'
        img_filename = 'twitter_hateful_img_DT.pdf'
        csv_sample_filename = 'twitter_hateful_sample_comments_DT.csv'
        txt_filename =  'twitter_hateful_tree.txt'
    if focus_label=='normal':
        clf_current = clf_normal
        clf_filename = 'twitter_normal_clf_DT.sav'
        img_filename = 'twitter_normal_img_DT.pdf'
        csv_sample_filename = 'twitter_normal_sample_comments_DT.csv'
        txt_filename =  'twitter_normal_tree.txt'
    if focus_label == 'spam':
        clf_current = clf_spam
        clf_filename = 'twitter_spam_clf_DT.sav'
        img_filename = 'twitter_spam_img_DT.pdf'
        csv_sample_filename = 'twitter_spam_sample_comments_DT.csv'
        txt_filename =  'twitter_spam_tree.txt'

    print("Focus label:", focus_label)
    print("pipeline:", [name for name, _ in clf_current.steps])
    print(clf_current['clf'])
    t0 = time()
    
    if mode== "train":
        print("Training model...")
        clf_current = clf_current.fit(agg_comments_train,agg_labels_train_encoded.ravel())
        print("Fit completed.")
        # save the model to disk
        joblib.dump(clf_current, clf_filename)
    else:
        print("Loading model")
        clf_current = joblib.load(clf_filename)
        print("Finished loading model.")
    
    print("Predicting...")        
    predicted = clf_current.predict(agg_comments_dev)
    print("Finished predicting.")
    
    # predicted = predicted.reshape(agg_labels_dev_encoded.shape)
    # print(predicted)
    f1_score_val = f1_score(agg_labels_dev_encoded, predicted, average='macro')
    precision_score_val = precision_score(agg_labels_dev_encoded, predicted, average='macro')
    recall_score_val =  recall_score(agg_labels_dev_encoded, predicted, average='macro')
    
    print("F1 score: ", f1_score_val)
    print("Precision score: ", precision_score_val)
    print("Recall score: ", recall_score_val)

    # print("comparing")
    # for real_label, predicted_label in zip(agg_labels_dev_encoded, predicted):
    #     print(real_label, predicted_label)
      
#%%
    
features_ = clf_current[0].get_feature_names()

text_title = "class: " + str(focus_label) + "; pos/neg features taken: " + str(n_display_values) +"\n"
text_title = text_title + "max_depth: "+ str(max_depth_var) + "\n"
text_title = text_title + "F1 score: " + "{:.3f}".format(f1_score_val)
text_title = text_title + "; Precision score: " + "{:.3f}".format(precision_score_val)
text_title = text_title + "; Recall score: " + "{:.3f}".format(recall_score_val)

text_representation = tree.export_text(clf_current[1],feature_names = features_)
text_representation = text_title + '\n' + text_representation 
print(text_representation)
with open(txt_filename, 'w') as f:
    f.write(text_representation)

#%%

fig = plt.figure(figsize=(100, 60))

fig.suptitle(text_title, fontsize=20, fontweight='bold')
tree.plot_tree(clf_current[1], 
                   feature_names=features_,  
                   class_names=["OTHER", focus_label],
                   filled=True,
                   #max_depth = 5,
                   fontsize=14)
fig.savefig(img_filename, bbox_inches = 'tight', dpi = 300)
