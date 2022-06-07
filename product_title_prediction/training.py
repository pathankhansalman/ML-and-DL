"""
Module for data analysis and training the data

@author: salman
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xgboost as xgb
import seaborn as sns
import os
import logging
import pickle
# This base_dir needs to be changed to re-run the code
base_dir = 'C:\\Users\\patha\\Downloads\\DS - assignment\\assignment\\'
os.chdir(base_dir)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from gensim.models import Word2Vec
# This is the local module with utility functions
from preprocessing import title_cleanup, drop_words
from sklearn.manifold import TSNE


logger = logging.getLogger('category_classification_training')
logging.basicConfig(level=logging.INFO)
# This model name needs to be run with 'NB', 'NN' and 'XGB'
model_name = 'XGB'

# Loading the data
# train.csv should be present in the folder above
filename = 'train.csv'
dfdata = pd.read_csv(base_dir + filename)

# Exploratory data analysis

# Setting this to False will suppress the outputs. I provided the observations
# in the comments. If required, this can be set to True to see all the outputs
verbose = False

# Counts
if verbose:
    print(dfdata.count())
# This shows that L3 category name is present for only 1/3 of the training data
# Since this is anyway only a "Good to have", we'll drop it
dfdata = dfdata.drop(columns=['l3_category_name'])


# Next, we intend to see if l1 and l2 category names are good enough proxies
# for category_id
cat_count = dfdata.groupby('category_id')['title'].count()
if verbose:
    print(len(dfdata['l1_category_name'].unique()))  # 13
    print(len(dfdata['l2_category_name'].unique()))  # 24
    print(dfdata[['l1_category_name', 'category_id']].drop_duplicates().
          sort_values(by='category_id'))
    print(dfdata[['l2_category_name', 'category_id']].drop_duplicates().
          sort_values(by='category_id'))
    # This shows that l2 is almost same as category_id, with 9, 10 and 14, 15
    # roll up into two L2 categories instead of 4
    # More importantly, category_id rolls up to l2 and l1, and no category id
    # spans across two different l1 categories
    ser_l1_count = dfdata.groupby('l1_category_name')['title'].count()
    plt.figure()
    ser_l1_count.plot(kind='bar', title='Product count by L1 category')
    plt.xticks(rotation=45, ha='right')
    plt.xlabel('L1 category')
    plt.ylabel('Count')
    plt.savefig(base_dir + 'l1_cat_product_count.png', bbox_inches='tight')
    plt.close()
    plt.figure()
    cat_count.plot(kind='bar', title='Product count by category ID')
    plt.xlabel('Category ID')
    plt.ylabel('Count')
    plt.savefig(base_dir + 'category_id_product_count.png')
    plt.close()
# There is class imbalance, and hence we need to provide class weight
class_weights = cat_count.div(cat_count.sum()).rdiv(1)
class_weights = class_weights.div(class_weights.sum())

# We store the unique L1 categories for other exploratory analysis
unique_l1_categories = dfdata['l1_category_name'].unique()
logger.info('Data stats exploration complete')


logger.info('Text pre-processing started')
# Clean the "title" field before exploring it further
dfdata_cleaned = title_cleanup(dfdata)
if verbose:
    for l1_cat in unique_l1_categories:
        print(l1_cat, dfdata_cleaned.loc[dfdata_cleaned['l1_category_name'] ==
                                         l1_cat,
                                         'title'].str.len().describe())
    # The distributions of title length look similar across L1 categories,
    # so we'll study it only at a overall level


# We look at the overall distribution of the title length
plt.figure()
dfdata_cleaned['title'].str.len().plot(kind='hist',
                                       grid=None,
                                       bins=50,
                                       title='Distribution of title lengths')
plt.xlabel('Title length')
plt.savefig(base_dir + 'title_length_hist.png')
plt.close()

# These assertions ensure that our title cleanup functions work okay
orig_len = dfdata.loc[dfdata_cleaned.index, 'title'].str.len()
new_len = dfdata_cleaned.loc[dfdata_cleaned.index, 'title'].str.len()

# Assertion 1: Length only reduced, and hasn't increased for whatever reason
assert((orig_len >= new_len).all())
# Assertion 2: There are no 0 length titles after filtering
assert(new_len.min() > 0)

# We then analyze the counts of words
sercount = pd.Series(' '.join(dfdata_cleaned['title'].values).split(' ')).\
    value_counts()
if verbose:
    print(len(set(' '.join(dfdata_cleaned['title'].values).split(' '))))
    plt.figure()
    sercount.plot(kind='hist',
                  title='Word count histogram',
                  bins=50)
    plt.savefig(base_dir + 'word_count.png')
    plt.close()
    plt.figure()
    sercount[sercount <= 10].plot(kind='hist',
                                  title='Word count histogram (zoomed)',
                                  bins=10)
    plt.savefig(base_dir + 'word_count_zoomed.png')
    plt.close()
    # Analysis at a category level
    for l1_cat in unique_l1_categories:
        print(l1_cat,
              len(set(' '.join(dfdata_cleaned.
                               loc[dfdata_cleaned['l1_category_name'] ==
                                   l1_cat,
                                   'title'].values).split(' '))))
        sercount = pd.Series(' '.join(dfdata_cleaned.
                                      loc[dfdata_cleaned['l1_category_name'] ==
                                          l1_cat,
                                          'title'].values).split(' ')).\
            value_counts()
        print(l1_cat, sercount.loc[[x for x in sercount.index if len(x) == 1]])
    # Analysis at a cross-category level
    common_words = []
    for l1_cat_1 in unique_l1_categories:
        comm_list = []
        for l1_cat_2 in unique_l1_categories:
            if l1_cat_1 == l1_cat_2:
                comm_list.append(np.nan)
                continue
            comm_list.append(
                    len(set(' '.join(dfdata_cleaned.
                                     loc[dfdata_cleaned['l1_category_name'] ==
                                         l1_cat_1,
                                         'title'].values).split(' ')).
                        intersection(
                        set(' '.join(dfdata_cleaned.
                                     loc[dfdata_cleaned['l1_category_name'] ==
                                         l1_cat_2,
                                         'title'].values).split(' ')))))
        common_words.append(comm_list)
        plt.figure()
        sns.heatmap(pd.DataFrame(common_words, index=unique_l1_categories,
                                 columns=unique_l1_categories))
        plt.title('Common word count across categories')
        plt.xticks(rotation=45, ha='right')
        plt.savefig(base_dir + 'common_words.png', bbox_inches='tight')
        plt.close()
# Unique number of words vary greatly at a L1 category level
# Since there is quite a bit of overlap, need to run a classification at the
# overall level, as opposed to at a sub-sample level

# After this, we remove two sets of words:
# Words which occur only once in the training set
# For optimizing for memory
# Words of length 1
# As it is seen below that these are present across all categories, and
# hence removing them is not a loss for our model; optimizes memory usage
drop_words_set = set(sercount[sercount == 1].index).\
    union(set([x for x in sercount.index if len(x) == 1]))
# This is an apply operation, but only takes 136 ms
dfdata_cleaned = drop_words(dfdata_cleaned, drop_words_set)


# Here we check the price column. Specifically, we check if there is any
# relation between category_id, and price being extreme (0 or a large outlier)
if verbose:
    print(dfdata_cleaned.loc[dfdata_cleaned['price'] == 0, 'category_id'].
          value_counts())
    print(dfdata_cleaned.loc[dfdata_cleaned['price'] >
                             dfdata_cleaned['price'].quantile(0.999),
                             'category_id'].value_counts())
    # This shows that a price of 0 is spread across all the categories
    # While outliers are mostly only present in Men's Fashion and Toys & Games
    # This warrants the use of price as a feature
logger.info('Text pre-processing finished')


# 3 methods tried: TF-idf + NB, Word2Vec + ANN / XGB
X_train, X_cv, y_train, y_cv = train_test_split(
    dfdata_cleaned.loc[:, ['l1_category_name', 'price', 'title']],
    dfdata_cleaned['category_id'],
    test_size=0.25,
    random_state=42)
logger.info('Train + validation split done')

if model_name == 'NB':
    # TF-idf + NB
    vectorizer = TfidfVectorizer()
    # This only takes 820 ms
    nb_train_x = vectorizer.fit_transform(X_train['title']).toarray()
    # Takes 1 min 10s
    clf_nb = MultinomialNB().fit(nb_train_x, y_train)
    with open(base_dir + 'nb_model.pkl', 'wb') as fp:
        pickle.dump(clf_nb, fp)
    with open(base_dir + 'vectorizer.pkl', 'wb') as fp:
        pickle.dump(vectorizer, fp)
    # Naive Bayes by default has fit_prior=True, to handle the class imbalance
    
    # Evaluating CV performance
    # Takes 3 seconds
    y_cv_pred_nb = clf_nb.predict(vectorizer.transform(X_cv['title']).
                                  toarray())
    print(accuracy_score(y_cv, y_cv_pred_nb))
    # CV accuracy: 85.81%
    
    # We get the predicted class probabilities and plot for the predicted class
    cv_pred_proba_nb = clf_nb.\
        predict_proba(vectorizer.transform(X_cv['title']).toarray())
    if verbose:
        plt.figure()
        plt.hist(np.max(cv_pred_proba_nb, axis=1), bins=100)
        plt.title('Naive Bayes predicted class probability (max)')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.savefig(base_dir + 'nb_pred_proba.png')
        plt.close()
elif model_name == 'NN' or model_name == 'XGB':
    # Word2Vec + ANN
    # In the word embeddings models, since the dimensionality is low, we will
    # add price as a feature as well
    # Training Word2Vec
    w2vmodel = Word2Vec(sentences=[x.split(' ')
                                   for x in X_train['title'].values],
                        min_count=1)
    with open(base_dir + 'w2v.pkl', 'wb') as fp:
        pickle.dump(w2vmodel, fp)
    if verbose:
        tsne = TSNE(random_state=42)
        all_vector_matrix = w2vmodel.syn1neg
        all_vector_matrix_2d = tsne.fit_transform(all_vector_matrix)
        comp_list = w2vmodel.wv.most_similar('computer')
        comp_idx = [w2vmodel.wv.key_to_index[x[0]] for x in comp_list]
        dfcomp = pd.DataFrame(all_vector_matrix_2d[comp_idx],
                              columns=['Feature 1', 'Feature 2'])
        dfcomp['Item'] = 'Computer'
        shirt_list = w2vmodel.wv.most_similar('shirt')
        shirt_idx = [w2vmodel.wv.key_to_index[x[0]] for x in shirt_list]
        dfshirt = pd.DataFrame(all_vector_matrix_2d[shirt_idx],
                               columns=['Feature 1', 'Feature 2'])
        dfshirt['Item'] = 'Shirt'
        book_list = w2vmodel.wv.most_similar('watch')
        book_idx = [w2vmodel.wv.key_to_index[x[0]] for x in book_list]
        dfbook = pd.DataFrame(all_vector_matrix_2d[book_idx],
                              columns=['Feature 1', 'Feature 2'])
        dfbook['Item'] = 'Watch'
        dfitem = pd.concat([dfcomp, dfshirt, dfbook], axis=0)
        plt.figure()
        sns.scatterplot(x='Feature 1', y='Feature 2',
                        hue=dfitem['Item'].tolist(),
                        palette=sns.color_palette('hls', 3),
                        data=dfitem)
        plt.title('Word2Vec similar word clusters')
        plt.legend(loc='best')
        plt.savefig(base_dir + 'w2v.png', bbox_inches='tight')
        plt.close()
        
    # Pre-processing the embedding data to be used later
    nn_train = X_train['title'].apply(lambda x:
                                      np.nanmean([w2vmodel.wv.get_vector(y)
                                                  for y in x.split(' ')
                                                  if y in
                                                  w2vmodel.wv.key_to_index],
                                                 axis=0))
    nn_train = pd.DataFrame([list(x) for x in nn_train.values],
                            index=nn_train.index)
    # Adding price and scaling
    # Scaling is important so that large values don't interefere with the
    # weight calculation
    scaler = StandardScaler()
    nn_train['price'] = scaler.\
        fit_transform(X_train['price'].values.reshape(-1, 1)).reshape(-1, )
    with open(base_dir + 'scaler.pkl', 'wb') as fp:
        pickle.dump(scaler, fp)
    # Pre-processing the embedding data for CV
    # Since there can be words we don't have in training data, we use nanmean
    # and fillna later
    nn_cv = X_cv['title'].apply(lambda x:
                                np.nanmean([w2vmodel.wv.get_vector(y)
                                            for y in x.split(' ')
                                            if y in w2vmodel.wv.key_to_index],
                                           axis=0))
    # Dummy value to fillna
    nan_val = [0.0]*100
    nn_cv = nn_cv.fillna(0)
    nn_cv = pd.DataFrame([nan_val if type(x) == int else list(x)
                          for x in nn_cv.values], index=nn_cv.index)
    # Adding price and L1 category name features (encoding appropriately)
    nn_cv['price'] = scaler.transform(X_cv['price'].values.reshape(-1, 1)).\
        reshape(-1, )
    if model_name == 'NN':
        # Initializing the ANN
        ann = tf.keras.models.Sequential()
        # Adding the input layer and the first hidden layer
        ann.add(tf.keras.layers.Dense(units=nn_train.shape[1],
                                      activation='relu'))
        # Adding the second hidden layer
        ann.add(tf.keras.layers.Dense(units=nn_train.shape[1],
                                      activation='relu'))
        # Adding the output layer
        ann.add(tf.keras.layers.Dense(units=y_train.nunique(),
                                      activation='softmax'))
        # Compiling the ANN
        ann.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                    metrics=['accuracy'])
        # Training the ANN on the Training set
        ann.fit(nn_train.values, (y_train - 1).values, batch_size=4096,
                epochs=50, verbose=0)
        tf.keras.models.save_model(ann, base_dir)
        metrics = ann.evaluate(nn_cv.values, (y_cv - 1).values, verbose=None)
        if verbose:
            # Plotting accuracy vs epochs. Done once to pick the number of
            # epochs
            # plt.figure()
            # plt.plot(ann.history.history['accuracy'])
            # plt.title('Training accuracy by epoch')
            # plt.xlabel('Epochs')
            # plt.ylabel('Accuracy')
            # plt.savefig(base_dir + 'nn_train_accuracy.png')
            # plt.close()
            logger.info('Neural net CV accuracy %.2f' % 100*metrics[1])
            cv_pred_proba_nn = np.max(ann.predict(nn_cv.values), axis=1)
            # 93.64% CV accuracy with L1 category, 79.2% without
            plt.figure()
            plt.hist(cv_pred_proba_nn, bins=100)
            plt.title('Neural network predicted class probability (max)')
            plt.xlabel('Probability')
            plt.ylabel('Frequency')
            plt.savefig(base_dir + 'nn_pred_proba.png')
            plt.close()
    else:
        # Word2Vec + SVM (Too slow)
        # Word2Vec + XGB
        xgb_model = xgb.XGBRFClassifier(use_label_encoder=False)
        xgb_train = nn_train.copy()
        xgb_train['price'] = X_train['price'].copy()
        xgb_cv = nn_cv.copy()
        xgb_cv['price'] = X_cv['price'].copy()
        xgb_model.fit(xgb_train, (y_train - 1))
        with open(base_dir + 'xgb_model.pkl', 'wb') as fp:
            pickle.dump(xgb_model, fp)
        y_cv_pred = xgb_model.predict(xgb_cv)
        cv_pred_proba_xgb = np.max(xgb_model.predict_proba(nn_cv), axis=1)
        if verbose:
            logger.info('XGB CV accuracy %.2f' %
                        100*accuracy_score((y_cv - 1), y_cv_pred))
            # 92.67% with L1 category, 76.07% without
            plt.figure()
            plt.hist(cv_pred_proba_xgb, bins=100)
            plt.title('XGBoost predicted class probability (max)')
            plt.xlabel('Probability')
            plt.ylabel('Frequency')
            plt.savefig(base_dir + 'xgb_pred_proba.png')
            plt.close()
logger.info('Model training done')
# Future research: n-gram analysis, KD Tree, SVM
