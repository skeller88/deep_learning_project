from typing import List

import numpy as np
import pandas as pd


def balanced_class_train_test_splits(*dataframes: List[pd.DataFrame]):
    train_dfs = []
    valid_dfs = []
    test_dfs = []

    train_size = .8
    for dataframe in dataframes:
        random_mask = np.random.rand(len(dataframe))
        train_mask = random_mask < train_size
        valid_mask = (train_size <= random_mask) & (random_mask < .9)
        test_mask = random_mask >= .9

        train_df = dataframe[train_mask]
        valid_df = dataframe[valid_mask]
        test_df = dataframe[test_mask]

        train_dfs.append(train_df)
        valid_dfs.append(valid_df)
        test_dfs.append(test_df)

        assert len(dataframe) == len(train_df) + len(test_df) + len(valid_df)
        for df in [train_df, valid_df, test_df]:
            assert len(df[df.duplicated()]) == 0

    train = pd.concat(train_dfs)
    valid = pd.concat(valid_dfs)
    test = pd.concat(test_dfs)

    for df in [train, valid, test]:
        assert len(df[df.duplicated()]) == 0

    print("len(train)", len(train), "len(valid)", len(valid), "len(test)", len(test))
    assert len(train) + len(valid) + len(test) == sum(map(len, dataframes))
    return train, valid, test
