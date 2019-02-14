#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 18:15:08 2019

@author: shiming
"""

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit, StratifiedKFold
import xgboost as xgb
from joblib import dump # model persistence

file = r''
index_file = file.replace('.xlsx','_Index.xlsx')
xls = pd.ExcelFile(file)
sheets = xls.sheet_names
index_xls = pd.ExcelFile(index_file)


n_splits = 20
################
#config xgboost
#param = {'max_depth':2, 'eta':1, 'silent':1, 'objective':'binary:logistic'}

#model = xgb.XGBClassifier(silent=False, scale_pos_weight=1,learning_rate=0.01, colsample_bytree = 0.4,subsample = 0.8,objective='binary:logistic', n_estimators=100, reg_alpha = 0.3, max_depth=4, gamma=10)

################


## over sheets
for sheet_idx, sheet in enumerate(sheets):
   df = xls.parse(sheet)
   df_idx = index_xls.parse(sheet_idx)
   # msg: 'Processing data on sheet'
   
   # preprocessing data
   # 
   
   n_exp = df_idx.shape[0]

   ## over experiments
   var = df_idx.keys()
   
   for idx_exp in range(n_exp):
      # find input col indices
      input_var = var[(df_idx.iloc[0]==1) | (df_idx.iloc[0]==3)]
      ## over outcomes
      for outcome_var in var[df_idx.iloc[idx_exp]==2]:
         #this_df = df[]

         ## do feature selection if necessary

         
         ## over splits


         #kfold = StratifiedKFold(n_splits = n_splits, shuffle=False)
         shufflesplit = StratifiedShuffleSplit(n_splits = n_splits, random_state = 42, test_size = 0.2)
         
         this_X = df[input_var]
         this_y = df[outcome_var]
         for train_idx, test_idx in shufflesplit(this_X, this_y):
            X_train, X_test = df[train_idx], df[test_idx]
            y_train, y_test = df[test_idx], df[test_idx]
            eval_set = [(X_train, y_train), (X_test, y_test)]
            eval_metric = ["auc","error"]
            
            # Do model training and testing
            this_model = model.fit(eval_set, eval_metric)
            
            dump(this_model, 'filename.joblib') 
## do model-based feature selection if necessary


## collect all models' performance; persistentize results
## do model selection



