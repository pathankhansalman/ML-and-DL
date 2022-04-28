"""
Created on Thu Apr 28 11:52:37 2022

@author: salman
"""
# Importing modules
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, StratifiedKFold,\
    GridSearchCV, cross_val_predict
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Reading the data
filename = 'C:\\Users\\salman\\Downloads\\creditcard\\creditcard.csv'
dfdata = pd.read_csv(filename)
run_model = 3
# Model numbers are as follows:
# 1 - XGB benchmark
# 2 - XGB CV
# 3 - SVM benchmark
# TODO:
# 4 - SVM CV
# 5 - PCA/Important features + XGB CV
# 6 - PCA/Important features + SVM CV

# Splitting classes, splitting each class further into train and test
# and concat (because of imbalance, need to ensure split)
df_pos = dfdata.loc[dfdata['Class'] == 1, :].iloc[:, :-1].copy()
y_pos = dfdata.loc[dfdata['Class'] == 1, ['Class']].copy()
df_neg = dfdata.loc[dfdata['Class'] == 0, :].iloc[:, :-1].copy()
y_neg = dfdata.loc[dfdata['Class'] == 0, ['Class']].copy()
X_train_pos, X_test_pos, y_train_pos, y_test_pos =\
    train_test_split(df_pos, y_pos, random_state=42)
X_train_neg, X_test_neg, y_train_neg, y_test_neg =\
    train_test_split(df_neg, y_neg, random_state=42)
X_train = pd.concat([X_train_pos, X_train_neg], axis=0)
X_test = pd.concat([X_test_pos, X_test_neg], axis=0)
y_train = pd.concat([y_train_pos, y_train_neg], axis=0)
y_test = pd.concat([y_test_pos, y_test_neg], axis=0)

if run_model == 1:
    # Model 1: XGBoost benchmark
    model_1 = xgb.XGBRFClassifier()
    model_1.fit(X_train, y_train.iloc[:, 0])
    y_pred = model_1.predict(dfdata.iloc[:, :-1])
    print(f1_score(dfdata.iloc[:, -1], y_pred))
    print(roc_auc_score(dfdata.iloc[:, -1], y_pred))
    # F1 score of 0.7757, AUC of 0.8373 (test)
    # F1 score of 0.8129, AUC of 0.8577 (overall)
elif run_model == 2:
    # Model 2: Tuned XGBoost
    model_2 = xgb.XGBRFClassifier()
    param_grid = {
      'scale_pos_weight': [1, 5, 10, 50, 100],
      'max_depth': [5, 10]
    }
    # StratifiedKFold to ensure split of classes in train/test
    shuffle = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(estimator=model_2, param_grid=param_grid,
                        n_jobs=-1, cv=shuffle, scoring='roc_auc')
    full_pred = cross_val_predict(grid, dfdata.iloc[:, :-1], dfdata['Class'],
                                  cv=shuffle, n_jobs=-1, verbose=3)
    print(f1_score(dfdata.iloc[:, -1], full_pred))
    print(roc_auc_score(dfdata.iloc[:, -1], full_pred))
    # F1 score of 0.7481, AUC of 0.9011 (overall)
elif run_model == 3:
    # Model 3: SVM benchmark
    model_3 = make_pipeline(StandardScaler(), SVC())
    model_3.fit(X_train, y_train.iloc[:, 0])
    y_pred = model_3.predict(dfdata.iloc[:, :-1])
    print(f1_score(dfdata.iloc[:, -1], y_pred))
    print(roc_auc_score(dfdata.iloc[:, -1], y_pred))
    # F1 score of 0.8762, AUC of 0.8994 (overall)
else:
    print('Model not implemented!')
