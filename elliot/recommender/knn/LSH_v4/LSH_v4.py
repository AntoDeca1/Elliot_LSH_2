import numpy as np
from ..LSH_v4.utils import stringify_array
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import time
import scipy.sparse as sp
from sklearn.metrics import pairwise_distances
import sys

"""
Implementation that instead follows Faiss' but without calculating similarities within the LSH class
1) For each vector I take the candidates in this way
 1a) I calculate the hamming distance between the hash of the vector and all other hashes in each table
 2a) I take all candidates
 3a) If I have more than k I take k random ones (according to the LSH pinecone article) the approximation is here
 Taken from the article : https://www.pinecone.io/learn/series/faiss/locality-sensitive-hashing-random-projection/
 "A single bucket containing 172_039 vectors. That means that we are choosing our top k values "at random" from those 172K vectors. 
 Clearly, we need to reduce our bucket size." 
"""


class RandomProjections():
    def __init__(self, d, nbits, l=1, seed=42):
        """
        :param d: Dimensionality of our original vectors (e.g number of users in the dataset)
        :param nbits: Number of hyperplanes
        :param l: Number of threes in the forest
        :self.all_hashes : The hashes of the buckets that contain something in
        :param seed:
        """
        self.nbits = nbits
        self.d = d
        self.l = l
        self.projection_matrix = self._initialize_projection_matrix()
        self.seed = seed
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
        buckets = self.project_matrix(input_matrix)
        self.buckets_matrix = (buckets > 0).astype(int)
        self.all_hashes = self.extract_unique_hashes()
        self.mapping_ = self.create_mappings()

    def _get_vec_candidates_new(self, vec, k):
        """
        For each vector pick his candidates
        This function uses hamming distance to pick the closest candidates
        :param vec: shape(n_tables,nbits)
        :return:
        """
        candidates = set()
        i = 0
        num_candidates = 0
        # For each vector the closest buckets indices in term of hamming dist
        closest_buckets_idxs = [self.hamming(vectors, table_id) for table_id, vectors in enumerate(vec)]
        while True:
            new_candidates = set()
            new_candidates_len = 0
            for index, table in enumerate(self.mapping_):
                closest_bucket = closest_buckets_idxs[index][i]
                new_candidates.update(table[stringify_array(self.all_hashes[index][closest_bucket])])
            effective_new_candidates = new_candidates.difference(candidates)
            new_candidates_len += len(effective_new_candidates)
            if num_candidates + new_candidates_len >= k:
                candidates = candidates | set(
                    np.random.choice(list(effective_new_candidates), (k - num_candidates), replace=False))
                break
            else:
                candidates = candidates | set(new_candidates)
                num_candidates = len(candidates)
                i += 1
        return candidates

    def _get_vec_candidates(self, vec, k):
        """
        For each vector pick his candidates
        This function uses hamming distance to pick the closest candidates
        :param vec: shape(n_tables,nbits)
        :return:
        """
        candidates = set()
        i = 0
        num_candidates = 0
        # For each vector the closest buckets indices in term of hamming dist
        closest_buckets_idxs = [self.hamming(vectors, table_id) for table_id, vectors in enumerate(vec)]
        while True:
            new_candidates = []
            new_candidates_len = 0
            for index, table in enumerate(self.mapping_):
                closest_bucket = closest_buckets_idxs[index][i]
                new_candidates.extend(table[stringify_array(self.all_hashes[index][closest_bucket])])
                new_candidates_len += len(new_candidates)
            if num_candidates + new_candidates_len >= k:
                candidates = candidates | set(np.random.choice(new_candidates, (k - num_candidates), replace=False))
                break
            else:
                candidates = candidates | set(new_candidates)
                num_candidates += len(new_candidates)
                i += 1
        return candidates

    def search(self, k):
        """
        Return a sparse matrix of shape n_itemsXn_items(n_usersXn_users) having only the candidate indexes set to 1
        :param k:
        :return:
        """
        n = len(self.buckets_matrix)
        output_matrix = np.zeros((n, n), dtype=int)
        for index, el in enumerate(self.buckets_matrix):
            candidates = list(self._get_vec_candidates_new(el, k))
            output_matrix[index, candidates] = 1
        # N.B Non passare alla scipy matrix rende il tutto piu rapido
        # return output_matrix
        return sp.csr_matrix(output_matrix)

    def search_2(self, k):
        """
        Instead of returning a sparse matrix n_itemsXn_items with only the candidates filled simply returns a matrix of shape
        n_itemsXcandidates (n_userXcandidates)
        :param k:
        :return:
        """
        n = len(self.buckets_matrix)
        candidates = np.zeros((n, k), dtype=int)
        for index, el in enumerate(self.buckets_matrix):
            candidates[index] = list(self._get_vec_candidates_new(el, k))
        return candidates

    def extract_unique_hashes(self):
        return {index: np.unique(el, axis=0) for index, el in enumerate(self.buckets_matrix.transpose(1, 0, 2))}

    def create_mappings(self):
        """
        For each bucket, it saves the list of elements that fell into it
        :return:
        """
        hash_tables = [defaultdict(list) for _ in range(self.l)]
        for item_idx, buckets in enumerate(self.buckets_matrix):
            for hash_table_id, bucket in enumerate(buckets):
                strigified_id = stringify_array(bucket)
                current_hash_table = hash_tables[hash_table_id]
                current_hash_table[strigified_id].append(item_idx)
        return hash_tables

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

    def hamming(self, hashed_vec: np.array, table_id: int) -> np.array:
        """
        Returns the matrix of buckets ordered relatively to the hamming distance
        :param hashed_vec: The bucket assigned to the vector we are considering
        :param other_hashes: All the buckets that have something in it
        :return: Matrix identical to "other_hashes" but ordered relatively to the hamming distance from the current hashed_vec
        Provare a calcolarle in un colpo solo per tutti i vettori
        """
        # get hamming distance between query vec and all buckets in other_hashes
        hamming_dist = np.count_nonzero(hashed_vec != self.all_hashes[table_id], axis=1)
        # Indices interal to self.all_hashes[table_id]
        sorted_indices = hamming_dist.argsort()
        return sorted_indices

    def _initialize_projection_matrix(self):
        """
        Useful to inizialize a matrix for projecting our dense vectors in binary ones
        :return:
        """
        # return np.random.randn(self.l, self.d, self.nbits)
        return np.random.rand(self.l, self.d, self.nbits) - .5
