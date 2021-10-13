import numpy as np
import pandas as pd
from zlib import crc32
from scipy.sparse.construct import rand
from sklearn.model_selection import train_test_split


def split_train_test(data, test_ratio):
    """split data set dataframe based on test ratio
    :param data: pandas dataframe
    :param test_ratio: test ratio to split by
    :return: dataframe of training set, test set
    """

    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio):
    """use the hash of each instance's identifier to determine if
    the hash is lower than or equal to the test_ratio of the maximum 
    hash value
    :param identifier: id to compute the hash from
    :param test_ratio: test ratio to split by
    :return:
    """

    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32


def split_train_test_by_id(data, test_ratio, id_column):
    """train/test split using the hash method
    :param data: dataset dataframe
    :param test_ratio: test ratio to split by
    :param id_column: column name of dataset object identifier
    :return: dataframe of training set, test set
    """

    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]




if __name__ == '__main__':
    pass
    
    housing = pd.read_csv('datasets/housing/housing.csv')
    train_set, test_set = split_train_test(housing, 0.2)
    assert np.allclose([len(train_set), len(test_set)], [16512, 4128])
    print(len(train_set), len(test_set))

    housing_with_id = housing.reset_index()
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'index')
    assert np.allclose([len(train_set), len(test_set)], [16512, 4128])
    print(len(train_set), len(test_set))
    housing_with_id['id'] = housing['longitude'] * 1000 + housing['latitude']
    train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, 'id')
    assert np.allclose([len(train_set), len(test_set)], [16322, 4318])
    print(len(train_set), len(test_set))

    # sklearn
    train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
    assert np.allclose([len(train_set), len(test_set)], [16512, 4128])
    print(len(train_set), len(test_set))
