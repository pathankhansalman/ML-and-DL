# product_title_prediction
This works on predicting the correct category for a product given its title and price.

In my use case, this takes a training set of about 140k products (further dividing into training + CV of 110k + 30k) and then tried to predict the categories for a test set (about 45k products).

However, this can be used with any other dataset, given the column names match.

## Overview of the solution

The solution is as follows:

1. Pre-process the text by removing extra spaces, punctuation and correcting the case
2. Further pre-process by removing single character words, words used only once in the entire training set
3. Vectorize the text data (by Tf-idf) or convert into embeddings (using Word2Vec)
4. Train models on top of the title vectors (Naive Bayes on Tf-idf and Neural network/XG Boost on Word2Vec)

### Tf-idf + NB: 88.28% training, 85.81% CV
### W2V + NN: 78.41% training, 77.03% CV
### W2V + XGB: 78.34% training, 76.08% CV
