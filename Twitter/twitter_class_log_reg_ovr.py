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

mode = "test"
focus_label = 'spam'

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

# to save model import joblib
import joblib 

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

    

    if focus_label=='abusive':
        clf_current = clf_abusive
        clf_filename = 'twitter_abusive_clf.sav'
        img_filename = 'twitter_abusive_img.pdf'
        csv_sample_filename = 'twitter_abusive_sample_comments.csv'
    if focus_label=='hateful':
        clf_current = clf_hateful
        clf_filename = 'twitter_hateful_clf.sav'
        img_filename = 'twitter_hateful_img.pdf'
        csv_sample_filename = 'twitter_hateful_sample_comments.csv'
    if focus_label=='normal':
        clf_current = clf_normal
        clf_filename = 'twitter_normal_clf.sav'
        img_filename = 'twitter_normal_img.pdf'
        csv_sample_filename = 'twitter_normal_sample_comments.csv'
    if focus_label == 'spam':
        clf_current = clf_spam
        clf_filename = 'twitter_spam_clf.sav'
        img_filename = 'twitter_spam_img.pdf'
        csv_sample_filename = 'twitter_spam_sample_comments.csv'

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
    
    # print("Predicting...")        
    # predicted = clf_current.predict(agg_comments_dev)
    # print("Finished predicting.")
    # predicted = predicted.reshape(agg_labels_dev_encoded.shape)
    # print(predicted)
    
    # print("Test F1 score: ", f1_score(agg_labels_dev_encoded, predicted, average='macro'))

    # print("comparing")
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


n_display_values = 15

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
#importance = most_neg + most_pred[::-1]
importance = most_pred[::-1]
#print(importance)

fig, ax = pyplot.subplots()
ax.tick_params(axis='both', which='major', labelsize=16)
pyplot.title(focus_label,fontdict = {'fontsize' : 16})
ax.bar([repr(x[1])[1:-1] for x in importance], [x[0] for x in importance], -.9, 0,  align='edge')
pyplot.xticks(rotation=90, ha='right')
#pyplot.show()

fig.tight_layout()
fig.savefig(img_filename,dpi=300)
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
# if(focus_label == 'abusive'):
#     print("Creating coefficients file, please go through all the other focus labels")
#     joined_df = pd.concat([most_neg_df, most_pred_df], axis=1, sort=False)
#     joined_df.to_csv('abusive_coefficients.csv')
    
# else:
#     print("Adding current model's coefficients and ngram to csv file")
#     coef_csv = pd.read_csv('abusive_coefficients.csv',index_col = 0)
#     joined_df = pd.concat([coef_csv, most_neg_df, most_pred_df], axis=1, sort=False)
#     joined_df.to_csv('abusive_coefficients.csv')

#%%
"""Analysis of the negative class ngrams"""


#for item in most_neg:
sample_ngrams = []
sample_comments = []
sample_labels = []
ngrams_list = most_pred[0:4]
counter = 0
for ngram_item in ngrams_list:
    ngram = ngram_item[1]
    counter = 0;
    for comment,label in zip(agg_comments_train, agg_labels_original):
        if label == focus_label:
            if ngram in comment.lower(): 
                labeled_comment = comment
                ngram_start_index = comment.lower().find(ngram)   
                while ngram_start_index is not -1:
                    f_comment_part = labeled_comment[0:ngram_start_index]
                    labeled_ngram = '<ng>' + ngram + '</ng>'
                    s_comment_part = labeled_comment[ngram_start_index+len(ngram):]
                    labeled_comment = f_comment_part + labeled_ngram + s_comment_part
                    ngram_start_index = labeled_comment.lower().find(ngram, ngram_start_index+9+len(ngram))
                counter = counter +1      
                sample_ngrams.append(ngram)
                sample_comments.append(labeled_comment)
                sample_labels.append(label)
                print(counter, ' || "'+ngram+'" || ', labeled_comment, " || ", label)
                #print("\n")
                if counter == 10 :
                        break
            
sample_ngrams_df = pd.DataFrame(sample_ngrams)
sample_ngrams_df = sample_ngrams_df.rename(columns={0:"ngram"})
sample_comments_df= pd.DataFrame(sample_comments)
sample_comments_df = sample_comments_df.rename(columns={0:"comment"})
sample_labels_df = pd.DataFrame(sample_labels)
sample_labels_df = sample_labels_df.rename(columns={0:"label"})
pd_sample_list = pd.concat([sample_ngrams_df,sample_comments_df,sample_labels_df],axis =1) 

pd_sample_list.to_csv(csv_sample_filename)   
print(counter)