import os
import os.path
import numpy as np
from collections import defaultdict
from ..LSH_v4.utils import stringify_array


class HashTable:
    def __init__(self, hash_size, dim, seed=42):
        self.table = defaultdict(list)
        self.hash_size = hash_size
        self.projections = np.random.randn(self.hash_size, dim)
        self._mapping = defaultdict(list)

    def add(self, input_matrix):
        """
        Project and construct the buckets mapping
        :param input_matrix:
        :return:
        """

        hashes = self._project(input_matrix)
        for idx, row in enumerate(hashes):
            self._mapping[stringify_array(row)].append(idx)

    def _project(self, input_matrix):
        signatures = input_matrix.dot(self.projections.T)
        return (signatures > 0).astype(int)

    def query(self, vecs):
        """
        :param vecs: Query vecs
        :return:
        """
        buckets = self._project(vecs)
        candidates = [self._mapping[stringify_array(bucket)] for bucket in buckets]
        return candidates

