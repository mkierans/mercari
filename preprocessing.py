import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.plotly as py
import plotly.graph_objs as go
pd.options.display.max_colwidth = -1


def load_data():
    train = pd.read_csv('data/train.tsv', sep='\t')
    test = pd.read_csv('data/test.tsv', sep='\t')


    train['general_cat'], train['sub_cat1'], train['sub_cat2'],\
        train['sub_cat3'], train['sub_cat4'] = \
            zip(*train['category_name'].apply(lambda x: split_cat(x)))

    test['general_cat'], test['sub_cat1'], test['sub_cat2'],\
        test['sub_cat3'], test['sub_cat4'] = \
            zip(*test['category_name'].apply(lambda x: split_cat(x)))
    return train, test


def split_cat(text):
    try:
        split = text.split('/')
        if len(split) == 3:
            split.extend(["only 3 labels", "only 3 labels"])
        elif len(split) == 4:
            split.extend(["only 4 labels"])
        return split
    except:
        # choosing np.nan enables a get_dummies option to ignore / include missing values
        return (np.nan, np.nan, np.nan, np.nan, np.nan)


def reshape_onehot(train, test, column, prefix='_', dummy_na=False,
                   rel_thresh=1, only_rel_cols=True, verbose=False):
    all_data = pd.concat([train, test], axis=0)

    if only_rel_cols:
        train_counts = train[column].value_counts()
        thresh_train = set(train_counts.loc[train_counts >= rel_thresh].index)
        if verbose:
            print("{} of {} values pass threshold of {}"
                  .format(len(thresh_train), len(train_counts), rel_thresh))
        test_set = set(test[column].unique())
        not_useful_cols = thresh_train.symmetric_difference(test_set)
        all_data.loc[all_data[column].isin(not_useful_cols), column] = np.nan

    new_columns = pd.get_dummies(all_data[column], prefix=prefix,
                                 dummy_na=dummy_na)
    new_data = pd.concat([all_data.drop([column], axis=1), new_columns],
                         axis=1)
    new_train = new_data.loc[new_data.test_id.isnull()].drop(['test_id'],
                                                             axis=1)
    new_test = new_data.loc[new_data.train_id.isnull()].drop(['train_id'],
                                                             axis=1)
    return new_train, new_test


def plot_column_dist(train, test, column):
    data = pd.concat([train, test], axis=0)
    train_name = 'train_' + column
    test_name = 'test_' + column
    train_counts = train[column].value_counts()
    test_counts = test[column].value_counts()
    train_counts.name = train_name
    test_counts.name = test_name
    merged = pd.concat([train_counts, test_counts], axis=1).fillna(0)
    merged.sort_values(train_name, inplace=True, ascending=False)

    N = len(merged)
    ind = np.arange(N)
    width = 0.35  # the width of the bars

    fig, ax = plt.subplots()
    ax.bar(ind, merged[train_name].values, width, color='r')
    ax.bar(ind + width, merged[test_name].values, width, color='y')

    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(merged.index, rotation='vertical')
    return ax


plot_column_dist(train, test, 'brand_name')
