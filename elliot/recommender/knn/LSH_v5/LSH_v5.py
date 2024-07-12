import os

import numpy as np
import pandas as pd
from ..LSH_v4.utils import stringify_array
from collections import defaultdict
import multiprocessing as mp

"Multiprocessing implementation"


class RandomProjections():
    def __init__(self, d, nbits, l=1, seed=42, initialization="uniform"):
        self.nbits = nbits
        self.d = d
        self.l = l
        self.seed = seed
        self.projection_matrix = self._initialize_projection_matrix(initialization=initialization)
        self.all_hashes = None
        if self.seed is not None:
            np.random.seed(self.seed)
        self.buckets_matrix = None

    def add(self, input_matrix):
        """
        Inserts each vector into the corresponding bucket(s)
        We have to consider that we could have also a forest of hyperplanes and so each vector will be inserted in l buckets
        1)  self.buckets_matrix : Matrix having as row the bucket relating to the corresponding vector
        2)  self.mapping: an object containing the mappings between buckets and contained vectors
        3)  self.all_hashes : Contains only the ids of buckets with content and not all the possible 2^nbits buckets id
        4)  self.closest_buckets: Pre-Computed closest buckets for each bucket in term of hamming distance(SLOW THE INDEX BUT MAKES THE SEARCH FASTER)
        :param input_matrix: Matrix having as rows the vectors we want to bucketize
        """
        self._input_matrix = input_matrix
        buckets = self.project_matrix(self._input_matrix)
        self.buckets_matrix = (buckets > 0).astype("uint8")
        self.all_hashes = self.extract_unique_hashes()
        self.mapping_ = self.create_mappings()

    def project_matrix(self, input_matrix):
        """
        Project vectors in the hamming space
        :param input_matrix:
        :return:
        """
        output = np.empty((self.l, input_matrix.shape[0], self.nbits))
        for i in range(self.projection_matrix.shape[0]):
            temp = input_matrix.dot(self.projection_matrix[i])
            temp = np.expand_dims(temp, axis=0)
            output[i] = temp
        return output.transpose(1, 0, 2)

    def extract_unique_hashes(self):
        # Transpose and extract unique elements
        return {index: np.unique(el, axis=0) for index, el in enumerate(self.buckets_matrix.transpose(1, 0, 2))}

    def _get_vec_candidates(self, vec, k, seed=42):
        if seed is not None:
            np.random.seed(seed)
        candidates = set()
        i = 0
        num_candidates = 0
        closest_buckets_idxs = [self.hamming(vectors, table_id) for table_id, vectors in enumerate(vec)]

        while True:
            new_candidates = set()
            for index, table in enumerate(self.mapping_):
                closest_bucket = closest_buckets_idxs[index][i]
                new_candidates.update(table[stringify_array(self.all_hashes[index][closest_bucket])])
            effective_new_candidates = new_candidates.difference(candidates)
            new_candidates_len = len(effective_new_candidates)
            if num_candidates + new_candidates_len >= k:
                candidates.update(
                    np.random.choice(list(effective_new_candidates), (k - num_candidates), replace=False))
                break
            else:
                candidates.update(new_candidates)
                num_candidates = len(candidates)
                i += 1
        return list(candidates)

    def search_2(self, k):
        n = len(self.buckets_matrix)
        num_workers = os.cpu_count() // 2
        chunk_size = max(1, n // num_workers)  # Ensure at least one item per chunk

        # Split the data into chunks
        chunks = [self.buckets_matrix[i:i + chunk_size] for i in range(0, n, chunk_size)]

        # Create a pool of workers
        with mp.Pool(processes=num_workers) as pool:
            # Map the chunks to the worker pool
            results = [pool.apply_async(self._process_chunk, args=(chunk, k)) for chunk in chunks]

            # Collect the results from the pool
            candidates = [result.get() for result in results]

        return np.vstack(candidates)

    def _process_chunk(self, chunk, k):
        chunk_candidates = np.zeros((len(chunk), k), dtype=int)
        for index, el in enumerate(chunk):
            seed = self.seed + index
            chunk_candidates[index] = self._get_vec_candidates(el, k, seed)
        return chunk_candidates

    def create_mappings(self):
        hash_tables = [defaultdict(list) for _ in range(self.l)]
        for item_idx, buckets in enumerate(self.buckets_matrix):
            for hash_table_id, bucket in enumerate(buckets):
                strigified_id = stringify_array(bucket)
                current_hash_table = hash_tables[hash_table_id]
                current_hash_table[strigified_id].append(item_idx)
        return hash_tables

    def project_matrix(self, input_matrix):
        output = np.empty((self.l, input_matrix.shape[0], self.nbits))
        for i in range(self.projection_matrix.shape[0]):
            temp = input_matrix.dot(self.projection_matrix[i])
            temp = np.expand_dims(temp, axis=0)
            output[i] = temp
        return output.transpose(1, 0, 2)

    def hamming(self, hashed_vec, table_id):
        hamming_dist = np.count_nonzero(hashed_vec != self.all_hashes[table_id], axis=1)
        sorted_indices = hamming_dist.argsort()
        return sorted_indices

    def _initialize_projection_matrix(self, initialization):
        print(initialization)
        if initialization == "gaussian":
            return np.random.normal(0, 1, (self.l, self.d, self.nbits))
        else:
            return np.random.rand(self.l, self.d, self.nbits) - 0.5
