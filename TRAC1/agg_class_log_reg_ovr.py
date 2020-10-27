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

mode = "test"
focus_label = 'OAG'

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

# to save model import joblib
import joblib 

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


if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    if focus_label=='NAG':
        clf_current = clf_NAG
        clf_filename = 'trac1_NAG_clf.sav'
        img_filename = 'trac1_NAG_img.pdf'
        img_filename_PN = 'trac1_NAG_img_PN.pdf'
        csv_sample_filename = 'trac1_NAG_sample_comments.csv'
        csv_sample_filename_P = 'trac1_NAG_sample_comments_P.csv'
        pvalues_txt_filename = 'trac1_NAG_pvalues.txt'
    if focus_label=='CAG':
        clf_current = clf_CAG
        clf_filename = 'trac1_CAG_clf.sav'
        img_filename = 'trac1_CAG_img.pdf'
        img_filename_PN = 'trac1_CAG_img_PN.pdf'
        csv_sample_filename = 'trac1_CAG_sample_comments.csv'
        csv_sample_filename_P = 'trac1_CAG_sample_comments_P.csv'
        pvalues_txt_filename = 'trac1_CAG_pvalues.txt'
    if focus_label=='OAG':
        clf_current = clf_OAG
        clf_filename = 'trac1_OAG_clf.sav'
        img_filename = 'trac1_OAG_img.pdf'
        img_filename_PN = 'trac1_OAG_img_PN.pdf'
        csv_sample_filename = 'trac1_OAG_sample_comments.csv'
        csv_sample_filename_P = 'trac1_OAG_sample_comments_P.csv'
        pvalues_txt_filename = 'trac1_OAG_pvalues.txt'

    print("Focus label:", focus_label)
    print("pipeline:", [name for name, _ in clf_current.steps])
    print(clf_current['clf'])
    t0 = time()
#%%   
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

n_display_values = 15


most_neg = neg_features[:n_display_values]
most_pred = predictive_features[:n_display_values]


#%%

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

"""Here we can select to save/print the most negative or 
most positive only using the importance variable"""
# get importance
importance = most_neg + most_pred[::-1]
#importance = most_pred[::-1]
#print(importance)

fig, ax = pyplot.subplots()
ax.tick_params(axis='both', which='major', labelsize=12)
pyplot.title(focus_label,fontdict = {'fontsize' : 12})
ax.bar([repr(x[1])[1:-1] for x in importance], [x[0] for x in importance], -.9, 0,  align='edge')
pyplot.xticks(rotation=90, ha='right')

#pyplot.show()



"""This line is used to save image only using positive coeficients"""
#fig.savefig(img_filename,dpi=300)
"""This line is used to save image using positive and negative coeficients"""
fig.savefig(img_filename_PN, dpi=300, bbox_inches='tight')

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
# if(focus_label == 'NAG'):
#     print("Creating coefficients file, please go through all the other focus labels")
#     joined_df = pd.concat([most_neg_df, most_pred_df], axis=1, sort=False)
#     joined_df.to_csv('trac1_coefficients.csv')
    
# else:
#     print("Adding current model's coefficients and ngram to csv file")
#     coef_csv = pd.read_csv('trac1_coefficients.csv',index_col = 0)
#     joined_df = pd.concat([coef_csv, most_neg_df, most_pred_df], axis=1, sort=False)
#     joined_df.to_csv('trac1_coefficients.csv')
        

#%% 
"""
Ahora se obtienen los p values
"""
from sklearn.feature_selection import chi2

X = clf_current[0].fit_transform(agg_comments_train)
scores, pvalues = chi2(X, agg_labels_train_encoded.ravel())

#%%

features_and_pvalues = list(zip(coefs[0],feature_names,pvalues,))

#%%
#sort using coefficient and then sort using p value
features_and_pvalues_neg = sorted(features_and_pvalues, 
                                  key=lambda x: x[0])# Most negative features

features_and_pvalues_neg = sorted(features_and_pvalues_neg, 
                                  key=lambda x: x[2])[15:30]

features_and_pvalues_neg_df = pd.DataFrame(features_and_pvalues_neg)
features_and_pvalues_neg_df = features_and_pvalues_neg_df.rename(columns={0:3, 1:4 , 2:5})
#%%
#sort using coefficient and then sort using p value
features_and_pvalues_pos = sorted(features_and_pvalues, 
                                  key=lambda x: x[0])
                                  #reverse = True)# Most positive features

features_and_pvalues_pos = sorted(features_and_pvalues_pos, 
                                  key=lambda x: x[2])[0:50]

features_and_pvalues_pos_df = pd.DataFrame(features_and_pvalues_pos)
#features_and_pvalues_pos_df = features_and_pvalues_pos_df.rename(columns={0:3, 1:4 , 2:5})

#%%

#features_and_pvalues_df = pd.concat([features_and_pvalues_pos_df], axis=1, sort=False)
features_and_pvalues_df = features_and_pvalues_pos_df

#features_and_pvalues_df.to_csv(pvalues_csv_filename)

features_and_pvalues_df = features_and_pvalues_df.round(4)

#%%%
print_counter = 0
with open(pvalues_txt_filename,'w') as f:
    print ('index & feature  & coefficient & p-value  \\\\')
    f.write('index & feature & coefficient & p-value  \\\\' + '\n')
    print ('\\hline')
    f.write('\\hline' + '\n')
    for index, row in features_and_pvalues_df.iterrows():
        text_line = ''
        if row[2] < 0.05:
            if row[2] < 0.001:
                pvalue_pos_txt = "\\bf{$<$ 0.001}"
            else:
                pvalue_pos_txt = str(row[2])
                pvalue_pos_txt = "\\bf{" + pvalue_pos_txt + "}"
        else:
            pvalue_pos_txt = str(row[2])
            
        text_line = text_line+'&' + str(index+1) + '&'+ '\say{'+ str(row[1])+'} '  +'& '+  str(row[0]) +'& ' + pvalue_pos_txt +' ' 
        text_line = text_line[1:] + '\\\\'
        f.write(text_line+'\n')
        f.write('\\hline' + '\n')
        if(print_counter < 15):
            print('\\hline')
            print(text_line)
        print_counter = print_counter + 1
    print(features_and_pvalues_df[0:15])
#%%
"""Analysis of the negative class ngrams"""


#for item in most_neg:
sample_ngrams = []
sample_comments = []
sample_labels = []
"""With the following line one can change what ngrams to look for in 
the comments: """
#ngrams_list = most_pred[0:4]
ngrams_list = features_and_pvalues_pos[0:10]

NAG_counter = 0
CAG_counter = 0
OAG_counter = 0
counter = 0

for ngram_item in ngrams_list:
    #ngram = ngram_item[1]
    ngram = ngram_item[1]
    
    NAG_counter = 0
    CAG_counter = 0
    OAG_counter = 0
    counter = 0;
    for comment,label in zip(agg_comments_train, agg_labels_original):
        
        if label == 'NAG':
            NAG_counter = NAG_counter +1
        if label == 'CAG':
            CAG_counter = CAG_counter +1
        if label == 'OAG':
            OAG_counter = OAG_counter +1
            
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
                #print(counter, ' || "'+ngram+'" || ', labeled_comment, " || ", label)
                #print("\n")
                if counter > 9 :
                        break
                    
#    print(ngram)
#    print("number of NAG comments:", NAG_counter)
#    print("number of CAG comments:", CAG_counter)
#    print("number of OAG comments:", OAG_counter)
            
sample_ngrams_df = pd.DataFrame(sample_ngrams)
sample_ngrams_df = sample_ngrams_df.rename(columns={0:"ngram"})
sample_comments_df= pd.DataFrame(sample_comments)
sample_comments_df = sample_comments_df.rename(columns={0:"comment"})
sample_labels_df = pd.DataFrame(sample_labels)
sample_labels_df = sample_labels_df.rename(columns={0:"label"})
pd_sample_list = pd.concat([sample_ngrams_df,sample_comments_df,sample_labels_df],axis =1) 
print(pd_sample_list)

"""Here change csv_sample_filename_P for pvalue ngrams or csv_sample_filename 
for coefficient ngrams"""
pd_sample_list.to_csv(csv_sample_filename_P)   
#print(counter)
#
#print("number of NAG comments:", NAG_counter)
#print("number of CAG comments:", CAG_counter)
#print("number of OAG comments:", OAG_counter)
#
#NAG_counter = 0
#CAG_counter = 0
#OAG_counter = 0
#
#for comment,label in zip(agg_comments_train, agg_labels_original):
#    if label == 'NAG':
#        NAG_counter = NAG_counter +1
#    if label == 'CAG':
#        CAG_counter = CAG_counter +1
#    if label == 'OAG':
#        OAG_counter = OAG_counter +1
#print("number of NAG comments:", NAG_counter)
#print("number of CAG comments:", CAG_counter)
#print("number of OAG comments:", OAG_counter)

