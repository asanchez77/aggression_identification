    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 17:23:19 2020

@author: 
    
"""


#%%

"""Load the data """

import os
import pandas as pd
import numpy as np

DATA_PATH = "data/"
"""
use mode = 'train' to train the model and 'test' to load a previously trained model
n_display_values is only valid when using 'train' mode
NAG:30
CAG:100
OAG:100
"""
mode = "test"
focus_label = 'OAG'
n_display_values = 100

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
    clf_log_reg_filename = 'trac1_NAG_clf.sav'
if focus_label=='CAG':
    clf_log_reg_filename = 'trac1_CAG_clf.sav'
if focus_label=='OAG':
    clf_log_reg_filename = 'trac1_OAG_clf.sav'

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


most_neg = neg_features[:n_display_values]
most_pred = predictive_features[:n_display_values]

select_feats =   most_pred + most_neg

#%%

vocab = {x[1]: i for i, x in enumerate(select_feats)}


#%%

clf_NAG = Pipeline([('tfidf', TfidfVectorizer(vocabulary=vocab) ),
              ('clf', DecisionTreeClassifier(random_state=1234,
                                             class_weight = 'balanced',
                                         ))
                ])

clf_CAG = Pipeline([('tfidf', TfidfVectorizer(vocabulary=vocab) ),
              ('clf', DecisionTreeClassifier(random_state=1234,
                                             class_weight = 'balanced',
                                         ))
                ])

clf_OAG = Pipeline([('tfidf', TfidfVectorizer(vocabulary=vocab) ),
              ('clf', DecisionTreeClassifier(random_state=1234,
                                         ))
                ])


if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    if focus_label=='NAG':
        clf_current = clf_NAG
        clf_filename = 'trac1_NAG_clf_DT.sav'
        img_filename = 'trac1_NAG_img_DT.pdf'
        csv_sample_filename = 'trac1_NAG_sample_comments_DT.csv'
        txt_filename =  'trac1_NAG_tree.txt'
    if focus_label=='CAG':
        clf_current = clf_CAG
        clf_filename = 'trac1_CAG_clf_DT.sav'
        img_filename = 'trac1_CAG_img_DT.pdf'
        csv_sample_filename = 'trac1_CAG_sample_comments_DT.csv'
        txt_filename =  'trac1_CAG_tree.txt'
    if focus_label=='OAG':
        clf_current = clf_OAG
        clf_filename = 'trac1_OAG_clf_DT.sav'
        img_filename = 'trac1_OAG_img_DT.pdf'
        csv_sample_filename = 'trac1_OAG_sample_comments_DT.csv'
        txt_filename =  'trac1_OAG_tree.txt'

    print("Focus label:", focus_label)
    print("pipeline:", [name for name, _ in clf_current.steps])
    print(clf_current['clf'])
    t0 = time()
#%%   0
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
    
    f1_score_val = f1_score(agg_labels_dev_encoded, predicted, average='macro')
    precision_score_val = precision_score(agg_labels_dev_encoded, predicted, average='macro')
    recall_score_val =  recall_score(agg_labels_dev_encoded, predicted, average='macro')
    print("F1 score: ", f1_score_val)
    print("Precision score: ", precision_score_val)
    print("Recall score: ", recall_score_val)
    #print("comparing")
    #for real_label, predicted_label in zip(agg_labels_dev_encoded, predicted):
        #print(real_label, predicted_label)
      
#%%

features_ = clf_current[0].get_feature_names()

text_title = "class: " + str(focus_label) + "; pos/neg features taken: " + str(n_display_values) +"\n"
text_title = text_title + "F1 score: " + "{:.3f}".format(f1_score_val)
text_title = text_title + "; Precision score: " + "{:.3f}".format(precision_score_val)
text_title = text_title + "; Recall score: " + "{:.3f}".format(recall_score_val)

text_representation = tree.export_text(clf_current[1], feature_names = features_)
text_representation = text_title + '\n' + text_representation 
print(text_representation)
with open(txt_filename, 'w') as f:
    f.write(text_representation)

#%%

fig = plt.figure(figsize=(100, 50))


fig.suptitle(text_title, fontsize=20, fontweight='bold')
tree.plot_tree(clf_current[1], 
                   feature_names=features_,  
                   class_names=["OTHER", focus_label],
                   filled=True,
                   #max_depth = 5,
                   fontsize=14)
fig.savefig(img_filename)



