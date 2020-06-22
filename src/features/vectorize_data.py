import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


def vectorize_data(clean_corpus, min_df=1, max_df=1.0, ngram_range=(1, 1)):
    """
    Creates a count vectorizer and a tfidf vectorizer
    :param clean_corpus: The corpus, typically as a series
    :param min_df: The minimum number of times a word needs to appear in the corpus to be included in the count vector (default 1)
    :param max_df: The maximum % of documents a word can appear in to be included in the vector (e.g. 1.0 means 100% of documents, default 1)
    :param ngram_range: The range of n_grams to be constructed by the vectorizer, as a tuple (default is (1,1))
    :return: The count vector and the tfidf vector
    """
    cv = CountVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    X = cv.fit_transform(clean_corpus)
    X = pd.DataFrame(X.toarray(), columns=cv.get_feature_names())
    cv_tfidf = TfidfVectorizer(min_df=min_df, max_df=max_df, ngram_range=ngram_range)
    X_tfidf = cv_tfidf.fit_transform(clean_corpus)
    X_tfidf = pd.DataFrame(X_tfidf.toarray(), columns=cv_tfidf.get_feature_names())
    return X, X_tfidf
