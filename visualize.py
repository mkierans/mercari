import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


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

# plot_column_dist(train, test, 'brand_name')
