import numpy as np
from scipy.sparse import csr_matrix


def create_sparse_matrix(rows, cols, sparsity=0.7, seed=42):
    np.random.seed(seed)
    dense_matrix = np.random.randint(1, 6, size=(rows, cols))
    mask = np.random.choice([False, True], size=dense_matrix.shape, p=[sparsity, 1 - sparsity])
    sparse_matrix = dense_matrix * mask
    return csr_matrix(sparse_matrix)


def count_valid_indices(my_list, index_list):
    """
    Given two lists count how many indexes i got correct
    :param my_list:
    :param index_list:
    :return:
    """
    valid_indices_count = sum([1 for index in my_list if index in index_list])
    return valid_indices_count / len(my_list)


def all_binary(n):
    total = 1 << n
    print(f"{total} possible combinations")
    combinations = []
    for i in range(total):
        # get binary representation of integer
        b = bin(i)[2:]
        # pad zeros to start of binary representtion
        b = '0' * (n - len(b)) + b
        b = [int(i) for i in b]
        combinations.append(b)
    return combinations


def stringify_array(array: np.array):
    return ''.join(map(str, array.flatten()))


def initialize_random_matrix(d, k):
    return np.random.rand(d, k)


def create_mappings(buckets):
    """
    Given the buckets assigned for each vector(supposing vector are number from 0 to n-1)
    :param buckets:
    :return:
    """
    mapping_ = {}
    for i in range(buckets.shape[0]):
        stringified_bucket_id = stringify_array(buckets[i, :])
        if stringified_bucket_id in mapping_.keys():
            mapping_[stringified_bucket_id].append(i)
        else:
            mapping_[stringified_bucket_id] = [i]
    return mapping_



