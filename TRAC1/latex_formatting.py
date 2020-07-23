#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 15:36:16 2020

@author: armando
"""

"""Load the data """

import os
import pandas as pd
import numpy as np

DATA_PATH = "."



def load_coefficient_data_file (csvfile, housing_path = DATA_PATH):
    csv_path = os.path.join(housing_path, csvfile)
    return pd.read_csv(csv_path,index_col = 0, skiprows=1, header= None)


coefficient_data = load_coefficient_data_file("trac1_coefficients.csv")
coefficient_data = coefficient_data.round(2)
num_columns = 4
init_column = 9

#%%
str_coefficient_data = coefficient_data.astype(str)
for index, row in str_coefficient_data.iterrows():
    text_line = ''
    for index_ in range(init_column,num_columns+init_column,2):
        text_line = text_line + '&'+ '\say{'+ row[index_+1]+'} ' +'& ' + row[index_] +' ' 
    text_line = text_line[1:] + '\\\\'
    print('\\hline')
    print(text_line)








