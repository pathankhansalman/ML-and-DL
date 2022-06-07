"""
Module for predicting the test category labels

@author: patha
"""
import pandas as pd
import numpy as np
import os
base_dir = 'C:\\Users\\patha\\Downloads\\DS - assignment\\assignment\\'
os.chdir(base_dir)
import logging
import pickle
from preprocessing import title_cleanup, drop_words
from tensorflow.keras.models import load_model

logger = logging.getLogger('category_classification_training')
logging.basicConfig(level=logging.INFO)
filename = 'test.csv'
dftest = pd.read_csv(base_dir + filename)
dftest = title_cleanup(dftest)
sercount = pd.Series(' '.join(dftest['title'].values).split(' ')).\
    value_counts()
drop_words_set = set(sercount[sercount == 1].index).\
    union(set([x for x in sercount.index if len(x) == 1]))
# This is an apply operation, but only takes 136 ms
dftest = drop_words(dftest, drop_words_set)
assert(dftest['title'].str.split(' ').str.len().min() > 0)
dfsubmission = pd.DataFrame()
dfsubmission['id'] = dftest['id'].copy()

# Top prediction: Naive Bayes
with open(base_dir + 'nb_model.pkl', 'rb') as fp:
    model_nb = pickle.load(fp)
with open(base_dir + 'vectorizer.pkl', 'rb') as fp:
    vectorizer = pickle.load(fp)
y_pred_test = model_nb.predict(vectorizer.transform(dftest['title']).
                             toarray())
dfsubmission['category_id_1'] = y_pred_test

# Second best prediction: Neural network
model_nn = load_model(base_dir)
with open(base_dir + 'w2v.pkl', 'rb') as fp:
    w2vmodel = pickle.load(fp)
with open(base_dir + 'scaler.pkl', 'rb') as fp:
    scaler = pickle.load(fp)
# Processing tht test titles same as CV (without fitting)
nn_test = dftest['title'].apply(lambda x:
                                np.nanmean([w2vmodel.wv.get_vector(y)
                                            for y in x.split(' ')
                                            if y in w2vmodel.wv.key_to_index],
                                           axis=0))
# Dummy value to fillna
nan_val = [0.0]*100
nn_test = nn_test.fillna(0)
nn_test = pd.DataFrame([nan_val if type(x) == int else list(x)
                      for x in nn_test.values], index=nn_test.index)
nn_test['price'] = scaler.transform(dftest['price'].values.reshape(-1, 1)).\
    reshape(-1, )
y_pred_test_nn = np.argmax(model_nn.predict(nn_test.values), axis=1) + 1
dfsubmission['category_id_2'] = y_pred_test_nn

# Third best prediction: XGBoost
with open(base_dir + 'xgb_model.pkl', 'rb') as fp:
    model_xgb = pickle.load(fp)
y_pred_test_xgb = model_xgb.predict(nn_test) + 1
dfsubmission['category_id_3'] = y_pred_test_xgb
dfsubmission.to_csv(base_dir + 'submission.csv', index=False)
