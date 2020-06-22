import pandas as pd
from pymongo import MongoClient


def balance(df):
    """
    Makes sure the sex of profiles is balanced in each age group. Original data has a 40/60 split female/male that
    lessens as the age groups increase
    :param df: Dataframe to balance by age group and sex
    :return: Balanced dataframe
    """
    groups = df.groupby(['age_group'])
    for count, group in enumerate(groups.groups):
        group = groups.get_group(group)
        sex_ratio = sum((group['sex'] == 'f') * 1) / sum((group['sex'] == 'm') * 1)
        if sex_ratio < 1:
            group_m = group[group['sex'] == 'm'].sample(frac=sex_ratio, random_state=1)
            group_f = group[group['sex'] == 'f']
        else:
            group_m = group[group['sex'] == 'm']
            group_f = group[group['sex'] == 'f'].sample(frac=(1 / sex_ratio), random_state=1)
        if count == 0:
            df_balanced = pd.concat([group_m, group_f], axis=0)
        else:
            df_balanced = pd.concat([df_balanced, group_m, group_f], axis=0)
    return df_balanced.sample(frac=1).reset_index(drop=True)


def import_data(column='_id', balance_data=True):
    """
    Imports data from mongoDB, adds some columns used in future analysis.
    :param column: Column we intend to include in the corpus.
    :param balance_data: Determines if we balance the data, so there are equal sexes in each age group (default True)
    :return: Dataframe of the data.
    """
    client = MongoClient()
    db = client['project4']
    collection = db['okc']
    df = pd.DataFrame(collection.find())
    df = df[~df[column].isnull()]
    age_filter = df['age'].astype(str).apply(lambda x: x.isdigit())
    df = df[age_filter]
    df['age'] = df['age'].astype(int)
    df = df[~df['sex'].isnull()].reset_index(drop=True)
    df['age_group'] = pd.qcut(df['age'], q=[0, .2, .4, .6, .8, 1])
    df['bi_age_group'] = '30 years or older'
    df.loc[df['age'] < 30, 'bi_age_group'] = '18 to 29 years old'
    df['pet_preference'] = ''
    prefer_cat = ['has cats', 'likes cats', 'dislikes dogs and has cats', 'dislikes dogs and likes cats',
                  'dislikes dogs']
    prefer_dog = ['likes dogs and dislikes cats', 'has dogs', 'has dogs and dislikes cats', 'likes dogs',
                  'dislikes cats']
    df.loc[df['pets'].isin(prefer_cat), 'pet_preference'] = 'cats'
    df.loc[df['pets'].isin(prefer_dog), 'pet_preference'] = 'dogs'
    if balance_data:
        df = balance(df)
    return df

