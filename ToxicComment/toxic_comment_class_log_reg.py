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

focus_label = "toxic"


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



#%%
from time import time
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

#%%

clf_toxic = Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', 
                                        ngram_range=(1, 5), lowercase=True) ),
              ('clf', LogisticRegression(penalty = 'l1',
                                         multi_class = 'ovr' ,
                                         solver='liblinear',
                                         C= 5.0,
                                         max_iter = 200))
                                         #))
                ])

clf_severe_toxic = Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', 
                                        ngram_range=(1, 4), lowercase=True) ),
              ('clf', LogisticRegression(penalty = 'l1',
                                         multi_class = 'ovr' ,
                                         solver='liblinear',
                                         C= 50.0,
                                         max_iter = 200))
                                         #))
                ])

clf_obscene = Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', 
                                        ngram_range=(2, 5), lowercase=True) ),
              ('clf', LogisticRegression(penalty = 'l1',
                                         multi_class = 'ovr' ,
                                         solver='liblinear',
                                         C= 5.0,
                                         max_iter = 200))
                                         #))
                ])

clf_threat = Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', 
                                        ngram_range=(2, 5), lowercase=True) ),
              ('clf', LogisticRegression(penalty = 'l1',
                                         multi_class = 'ovr' ,
                                         solver='liblinear',
                                         C= 10.0,
                                         max_iter = 200))
                                         #))
                ])

clf_insult = Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', 
                                        ngram_range=(1, 5), lowercase=True) ),
              ('clf', LogisticRegression(penalty = 'l1',
                                         multi_class = 'ovr' ,
                                         solver='liblinear',
                                         C= 5.0,
                                         max_iter = 200))
                                         #))
                ])

clf_identity_hate = Pipeline([('tfidf', TfidfVectorizer(binary=True, analyzer='char', 
                                        ngram_range=(1, 4), lowercase=True) ),
              ('clf', LogisticRegression(penalty = 'l1',
                                         multi_class = 'ovr' ,
                                         solver='liblinear',
                                         C= 10.0,
                                         max_iter = 200))
                                         #))
                ])

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    print("Training model...")

    if focus_label=='toxic':
        clf_current = clf_toxic
    if focus_label=='severe_toxic':
        clf_current = clf_severe_toxic
    if focus_label=='obscene':
        clf_current = clf_obscene
    if focus_label=='threat':
        clf_current = clf_threat
    if focus_label=='insult':
        clf_current = clf_insult
    if focus_label=='identity_hate':
        clf_current = clf_identity_hate
        
    print(focus_label)    

    print("pipeline:", [name for name, _ in clf_current.steps])
    print(clf_current['clf'])
    t0 = time()
    clf_current = clf_current.fit(agg_comments_train,agg_labels_train.ravel())
    print("Fit completed.")
    predicted = clf_current.predict(agg_comments_dev)

    predicted = predicted.reshape(agg_labels_dev.shape)
    print(predicted)
    
    print("F1 score: ", f1_score(agg_labels_dev, predicted, average='macro'))
    #print("comparing")
    #for real_label, predicted_label in zip(agg_labels_dev_encoded, predicted):
        #print(real_label, predicted_label)
      
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


n_display_values = 15

most_neg = neg_features[:n_display_values]
most_pred = predictive_features[:n_display_values]

def print_format_coef(features_coef):
    for feature in features_coef:
        repr_string = repr(feature[1])
        repr_string = repr_string[1:]
        repr_string = repr_string[:-1]
        print('(%.2f, "%s")' % (feature[0],repr_string))
    return


print("-------------")
print("most negative features")
print_format_coef(most_neg)
print("-------------")
print("most predictive features")
print_format_coef(most_pred)


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

n_list_values =  30
most_neg_list = neg_features[:n_list_values]
most_pred_list = predictive_features[:n_list_values]

most_neg_df =  pd.DataFrame(list(most_neg_list))
most_neg_df = most_neg_df.rename(columns={0:focus_label+"_neg_coef",1:focus_label+"_neg_ngram"})
most_pred_df =  pd.DataFrame(list(most_pred_list))
most_pred_df = most_pred_df.rename(columns={0:focus_label+"_pred_coef",1:focus_label+"_pred_ngram"})
#%%

"""If NAG focus label is used, it will create or overwrite the csv file, else 
it will open the existing file (it asumes it was previously created) and it
will add the next model's n-grams and coefficients
"""
if(focus_label == 'toxic'):
    print("Creating coefficients file, please go through all the other focus labels")
    joined_df = pd.concat([most_neg_df, most_pred_df], axis=1, sort=False)
    joined_df.to_csv('toxic_coefficients.csv')
    
else:
    print("Adding current model's coefficients and ngram to csv file")
    coef_csv = pd.read_csv('toxic_coefficients.csv',index_col = 0)
    joined_df = pd.concat([coef_csv, most_neg_df, most_pred_df], axis=1, sort=False)
    joined_df.to_csv('toxic_coefficients.csv')
        