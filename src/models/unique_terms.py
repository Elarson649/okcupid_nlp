def normcdf(x):
    """
    Creates a CDF out of x and returns where each value x lies in the CDF.
    :param x: A list of values
    :return: The CDF values of each value in x
    """
    from scipy.stats import norm
    return norm.cdf(x, x.mean(), x.std())


def unique_terms(count_vector, df, category='bi_age_group', b=.5):
    """
    Calculates the f-score for each word/term in the count_vector. Based on scattertext.
    :param count_vector: The count vector of the corpus.
    :param df: The original dataframe used before making the corpus
    :param category: The binary category we want to create F-scores for
    :param b: The beta value to use in the weighted harmonic mean (default .5, which weights precision higher than frequency)
    :return: A series of the terms and the f-scores, sorted
    """
    rows = count_vector.shape[0]-1
    count_vector[category] = df.loc[0:rows, category]
    count_vector = count_vector[count_vector[category].notnull()]
    # weigh how common the word is in a category vs all other categories more than the frequency in the given category
    scoring_matrix = count_vector.groupby(category).sum().T
    negative = scoring_matrix.columns[0]
    positive = scoring_matrix.columns[1]
    scoring_matrix['pos_precision'] = (scoring_matrix[positive] * 1.) / (scoring_matrix[positive] + scoring_matrix[negative])
    scoring_matrix['pos_freq_pct'] = (scoring_matrix[positive] * 1.) / (scoring_matrix[positive].sum())
    scoring_matrix['pos_precision_normcdf'] = normcdf(scoring_matrix.pos_precision)
    scoring_matrix['pos_freq_pct_normcdf'] = normcdf(scoring_matrix.pos_freq_pct.values)
    scoring_matrix['pos_scaled_f_score'] = ((1 + b ** 2) * scoring_matrix['pos_precision_normcdf'] * scoring_matrix['pos_freq_pct_normcdf']) / (
                (b ** 2) * scoring_matrix['pos_precision_normcdf'] + scoring_matrix['pos_freq_pct_normcdf'])
    scoring_matrix['neg_precision_normcdf'] = normcdf(
        (scoring_matrix[negative] * 1.) / (scoring_matrix[negative] + scoring_matrix[positive]))
    scoring_matrix['neg_freq_pct_normcdf'] = normcdf((scoring_matrix[negative] * 1.) / (scoring_matrix[negative].sum()))
    scoring_matrix['neg_scaled_f_score'] = ((1 + b ** 2) * scoring_matrix['neg_precision_normcdf'] * scoring_matrix['neg_freq_pct_normcdf']) / (
                (b ** 2) * scoring_matrix['neg_precision_normcdf'] + scoring_matrix['neg_freq_pct_normcdf'])
    scoring_matrix['scaled_f_score'] = 0
    scoring_matrix.loc[scoring_matrix['pos_scaled_f_score'] > scoring_matrix['neg_scaled_f_score'], 'scaled_f_score'] = scoring_matrix['pos_scaled_f_score']
    scoring_matrix.loc[scoring_matrix['pos_scaled_f_score'] < scoring_matrix['neg_scaled_f_score'], 'scaled_f_score'] = 1 - scoring_matrix['neg_scaled_f_score']
    scoring_matrix['scaled_f_score'] = 2 * (scoring_matrix['scaled_f_score'] - 0.5)
    return scoring_matrix.sort_values(by='scaled_f_score', ascending=False)