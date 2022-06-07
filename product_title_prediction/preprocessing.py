""" Module performing the text cleanup operations

@author: patha
"""
import re
import nltk
# These commands download the required packages from nltk
nltk.download('stopwords')
nltk.download('punkt')


def title_cleanup(dfdata, title_col='title'):
    def _title_cleanup_helper_(text):
        cleaned_text = re.sub("\\s+", " ", text)
        cleaned_text = re.sub("[^0-9A-Za-z ]", "", cleaned_text)
        cleaned_text = cleaned_text.lower()
        cleaned_text = nltk.tokenize.word_tokenize(cleaned_text)
        cleaned_text = [i for i in cleaned_text if i not in stopwords]
        # We are skipping lemmatization, as this is not a prose text
        return ' '.join(cleaned_text)
    stopwords = nltk.corpus.stopwords.words('english')
    dfdata_cleaned = dfdata.copy()
    # There is a nan at index 2399
    dfdata_cleaned = dfdata_cleaned.dropna(subset=['title']).copy()
    # Takes 8 seconds on the entire data
    dfdata_cleaned.loc[:, 'title'] = dfdata_cleaned['title'].astype(str).\
        apply(_title_cleanup_helper_)
    # This operation gives 256 rows with 0 title size
    # 254 of these are in foreign language, have only stop words,
    # have only a single character (letter or punctuation)
    # 2 of them, have some weird format, so I am re-adding them here
    dfdata_cleaned.loc[4909, 'title'] = 'focus kepada skin treatment'
    dfdata_cleaned.loc[12754, 'title'] =\
        'mixy beauty full bounce 3 1 lip kit 2 lipmattes 1 lipgloss'
    # Dropping the other 254 columns
    dfdata_cleaned = dfdata_cleaned.loc[dfdata_cleaned['title'].str.len() > 0,
                                        :].copy()
    return dfdata_cleaned


def drop_words(dfdata, drop_words_set):
    def _drop_words_helper_(text):
        return ' '.join([x for x in text.split(' ')
                         if x not in drop_words_set])
    dfdata_cleaned = dfdata.copy()
    dfdata_cleaned.loc[:, 'title'] = dfdata_cleaned['title'].astype(str).\
        apply(_drop_words_helper_)
    return dfdata_cleaned
