import gc
import pickle

import numpy as np
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, haversine_distances, chi2_kernel, \
    manhattan_distances
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import normalize
from ..LSH_HashTables.LSH import LSHRp as LSHHashTables
from ..LSH_v4.LSH_v4 import RandomProjections as CustomLSH
from ..LSH_faiss.LSH_faiss import RandomProjections as FaissLSH
from ..LSH_v3.LSH_v3 import RandomProjections as LSH_noHamming
import time
import scipy.sparse as sp
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Pool
from tqdm import tqdm
from collections import defaultdict


class Similarity(object):
    """
    Simple kNN class
    """

    def __init__(self, data, num_neighbors, similarity, implicit, nbits, ntables):
        self._data = data
        self._ratings = data.train_dict
        self._num_neighbors = num_neighbors
        self._similarity = similarity
        if self._similarity in ["rp_faiss", "rp_custom", "rp_hashtables"]:
            self._lsh_times_obj = {}
        else:
            self._lsh_times_obj = None
        self._implicit = implicit

        # NEW CODE HERE (LSH parameters)
        self._nbits = nbits
        self._ntables = ntables

        if self._implicit:
            self._URM = self._data.sp_i_train
        else:
            self._URM = self._data.sp_i_train_ratings

        self._users = self._data.users
        self._items = self._data.items
        self._private_users = self._data.private_users
        self._public_users = self._data.public_users
        self._private_items = self._data.private_items
        self._public_items = self._data.public_items

    def initialize(self):
        """
        This function initialize the data model
        """

        self.supported_similarities = ["cosine", "dot", ]
        self.supported_dissimilarities = ["euclidean", "manhattan", "haversine", "chi2", 'cityblock', 'l1', 'l2',
                                          'braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming',
                                          'jaccard', 'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto',
                                          'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
                                          'yule']
        print(f"\nSupported Similarities: {self.supported_similarities}")
        print(f"Supported Distances/Dissimilarities: {self.supported_dissimilarities}\n")

        print(f"Similarity used : {self._similarity} with {self._num_neighbors} neighbors")

        self._similarity_matrix = np.empty((len(self._items), len(self._items)))

        data, rows_indices, cols_indptr = [], [], []
        if self._similarity in ["rp_faiss", "rp_custom"]:
            data, rows_indices, cols_indptr = self.lsh_similarity(self._URM.T)
        else:
            self.process_similarity(self._similarity)

            column_row_index = np.arange(len(self._data.items), dtype=np.int32)

            for item_idx in range(len(self._data.items)):
                cols_indptr.append(len(data))
                column_data = self._similarity_matrix[item_idx, :]

                non_zero_data = column_data != 0
                idx_sorted = np.argsort(column_data[non_zero_data])  # sort by column
                top_k_idx = idx_sorted[-self._num_neighbors:]
                data.extend(column_data[non_zero_data][top_k_idx])
                rows_indices.extend(column_row_index[non_zero_data][top_k_idx])

            cols_indptr.append(len(data))

        W_sparse = sparse.csc_matrix((data, rows_indices, cols_indptr),
                                     shape=(len(self._data.items), len(self._data.items)), dtype=np.float32).tocsr()
        self._preds = self._URM.dot(W_sparse).toarray()

        del self._similarity_matrix

    def process_similarity(self, similarity):
        if similarity == "cosine":
            self._similarity_matrix = cosine_similarity(self._URM.T)
        elif similarity == "dot":
            self._similarity_matrix = (self._URM.T @ self._URM).toarray()
        elif similarity == "euclidean":
            self._similarity_matrix = (1 / (1 + euclidean_distances(self._URM.T)))
        elif similarity == "manhattan":
            self._similarity_matrix = (1 / (1 + manhattan_distances(self._URM.T)))
        elif similarity == "haversine":
            self._similarity_matrix = (1 / (1 + haversine_distances(self._URM.T)))
        elif similarity == "baseline":
            self._similarity_matrix = self.baseline(self._URM.T)
        elif similarity == "rp_hashtables":
            print(f"{self._similarity} similarity with nbits: {self._nbits} and ntables: {self._ntables} ")
            lsh_index = LSH_noHamming(d=len(self._users), nbits=self._nbits, l=self._ntables)
            prima = time.time()
            lsh_index.add(self._URM.T)
            indexing_time = time.time() - prima
            print(indexing_time, "Time to index the vectors")
            self._lsh_times_obj["indexing_time"] = indexing_time
            prima = time.time()
            candidates_matrix = lsh_index.search_2()
            candidates_retrieval_time = time.time() - prima
            print(candidates_retrieval_time, "Time to pull out the candidates")
            prima = time.time()
            self._similarity_matrix = self.compute_candidates_cosine_similarity(item_user_matrix=self._URM.T,
                                                                                candidate_matrix=candidates_matrix)
            similarity_matrix_time = time.time() - prima
            print(similarity_matrix_time, "Time for calculating the similarity matrix")
            self._lsh_times_obj["similarity_matrix_time"] = similarity_matrix_time

            self._lsh_times_obj["candidates_retrieval_time"] = candidates_retrieval_time
        elif similarity == "chi2":
            self._similarity_matrix = (1 / (1 + chi2_kernel(self._URM.T)))
        elif similarity in ['cityblock', 'l1', 'l2']:
            self._similarity_matrix = (1 / (1 + pairwise_distances(self._URM.T, metric=similarity)))
        elif similarity in ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard',
                            'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                            'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']:

            self._similarity_matrix = (1 / (1 + pairwise_distances(self._URM.T.toarray(), metric=similarity)))
        else:
            raise ValueError("Compute Similarity: value for parameter 'similarity' not recognized."
                             f"\nAllowed values are: {self.supported_similarities}, {self.supported_dissimilarities}."
                             f"\nPassed value was {similarity}\nTry with implementation: aiolli")

    def compute_candidates_cosine_similarity_mp(self, item_user_matrix, candidate_matrix):
        """
        For each item(in case of ItemKNN) pick its candidates and populate the corrisponding row in the similarity matrix
        :param item_user_matrix:
        :param candidate_matrix:
        :return:
        """
        item_user_matrix = item_user_matrix.tocsr()
        n_items = candidate_matrix.shape[0]
        # MULTIPROCESSING HERE POSSIBLY
        n_processes = mp.cpu_count() // 2
        chunk_size = int(np.ceil(n_items / n_processes))
        data, rows_indices = [], []
        cols_indptr = np.arange(0, (n_items * self._num_neighbors) + 1, self._num_neighbors)
        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            temp_data, temp_indices = [0] * n_processes, [0] * n_processes
            tasks = []
            for idx in range(0, n_processes):
                candidates_matrix_chunk = candidate_matrix[idx * chunk_size:(idx + 1) * chunk_size]
                tasks.append(executor.submit(compute_similarity, candidates_matrix_chunk, item_user_matrix, idx))
            for future in as_completed(tasks):
                data_res, rows_indices_res, chunk_idx = future.result()
                temp_data[chunk_idx] = data_res
                temp_indices[chunk_idx] = rows_indices_res
            for el_data, el_index in zip(temp_data, temp_indices):
                data.extend(el_data)
                rows_indices.extend(el_index)
        return data, rows_indices, cols_indptr

    def baseline(self, item_user_matrix):
        """
        In the case of Faiss or CustomLSH(my implementation of Faiss) we already retrieve the k candidates
        It makes no sense to sort them relatevely to the cosine similarity and so we skip this step and we output directly the components of the final sparse matrix
        :param item_user_matrix:
        :param candidate_matrix:
        :return:
        """
        item_user_matrix = item_user_matrix.tocsr()
        n_users = item_user_matrix.shape[0]
        similarity_matrix = np.empty((n_users, n_users))
        for i in range(n_users):
            # Compute cosine similarity between item i and its candidates
            item_vector = item_user_matrix[i]
            sim_scores = cosine_similarity(item_vector, item_user_matrix)
            similarity_matrix[i] = sim_scores
        return similarity_matrix

    def compute_candidates_cosine_similarity(self, item_user_matrix, candidate_matrix):
        """
        For each item(in case of ItemKNN) pick its candidates and populate the corrisponding row in the similarity matrix
        :param item_user_matrix:
        :param candidate_matrix:
        :return:
        """
        item_user_matrix = item_user_matrix.tocsr()
        n_items = candidate_matrix.shape[0]
        similarity_matrix = np.empty((len(self._items), len(self._items)))
        # MULTIPROCESSING HERE
        for i in range(n_items):
            # Get the indices of the candidates for the i-th item
            candidate_indices = candidate_matrix.getrow(i).nonzero()[1]

            # Extract the relevant vectors from URM for these candidates
            URM_candidates = item_user_matrix[candidate_indices, :]

            # Compute cosine similarity between item i and its candidates
            item_vector = item_user_matrix[i, :]
            sim_scores = cosine_similarity(item_vector, URM_candidates)

            # Store the results
            similarity_matrix[i, candidate_indices] = sim_scores
        return similarity_matrix

    def compute_candidates_cosine_similarity_fast(self, item_user_matrix, candidate_matrix):
        """
        In the case of Faiss or CustomLSH(my implementation of Faiss) we already retrieve the k candidates
        It makes no sense to sort them relatevely to the cosine similarity and so we skip this step and we output directly the components of the final sparse matrix
        :param item_user_matrix:
        :param candidate_matrix:
        :return:
        """
        item_user_matrix = item_user_matrix.tocsr()
        n_items = candidate_matrix.shape[0]
        data, rows_indices, cols_indptr = [], [], []
        for i in range(n_items):
            cols_indptr.append(len(data))
            # Get the indices of the candidates for the i-th item
            candidate_indices = candidate_matrix[i]

            # Extract the relevant vectors from URM for these candidates
            URM_candidates = item_user_matrix[candidate_indices, :]

            # Compute cosine similarity between item i and its candidates
            item_vector = item_user_matrix[i]
            sim_scores = cosine_similarity(item_vector, URM_candidates)
            data.extend(sim_scores.squeeze())
            rows_indices.extend(candidate_indices)
        cols_indptr.append(len(data))

        return data, rows_indices, cols_indptr

    def lsh_similarity(self, item_user_matrix):
        """
        1) Instantiate the Index
        2) Project vector in a lower dimensional space
        3) Pick the candidates
        4) Compute the cosine similarity starting from the candidates taken from LSH
        :param item_user_matrix:
        :return:
        """
        if self._similarity == "rp_faiss":
            print(f"{self._similarity} similarity with nbits: {self._nbits} and ntables: 1 ")
            rp = FaissLSH(d=len(self._users), nbits=self._nbits)
        else:
            print(f"{self._similarity} similarity with nbits: {self._nbits} and ntables: {self._ntables} ")
            rp = CustomLSH(d=len(self._users), nbits=self._nbits, l=self._ntables)
        prima = time.time()
        rp.add(item_user_matrix)
        indexing_time = time.time() - prima
        self._lsh_times_obj["indexing_time"] = indexing_time
        print(indexing_time, "Time to index the vectors")
        prima = time.time()
        if isinstance(rp, FaissLSH):
            candidates_matrix = rp.search_2(item_user_matrix, k=self._num_neighbors)
        else:
            candidates_matrix = rp.search_2(k=self._num_neighbors)
        candidates_retrieval_time = time.time() - prima
        print(candidates_retrieval_time, "Time to pull out the candidates")
        # The item_user_matrix
        self._lsh_times_obj["candidates_retrieval_time"] = candidates_retrieval_time
        # CHECK IF IT MAKES TO SENSE TO DO SO: Idea: Evoid this pattern in the similarity_matrix_time
        del rp
        # gc.collect()
        prima = time.time()
        data, rows_indices, cols_indptr = self.compute_candidates_cosine_similarity_fast(item_user_matrix,
                                                                                         candidates_matrix)
        similarity_matrix_time = time.time() - prima
        print(similarity_matrix_time, "Time for calculating the similarity matrix")
        self._lsh_times_obj["similarity_matrix_time"] = similarity_matrix_time
        return data, rows_indices, cols_indptr

    def get_user_recs(self, u, mask, k):
        user_id = self._data.public_users.get(u)
        user_recs = self._preds[user_id]
        # user_items = self._ratings[u].keys()
        user_recs_mask = mask[user_id]
        user_recs[~user_recs_mask] = -np.inf
        indices, values = zip(*[(self._data.private_items.get(u_list[0]), u_list[1])
                                for u_list in enumerate(user_recs)])

        # indices, values = zip(*predictions.items())
        indices = np.array(indices)
        values = np.array(values)
        local_k = min(k, len(values))
        partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
        real_values = values[partially_ordered_preds_indices]
        real_indices = indices[partially_ordered_preds_indices]
        local_top_k = real_values.argsort()[::-1]
        return [(real_indices[item], real_values[item]) for item in local_top_k]

    # @staticmethod
    # def score_item(neighs, user_items):
    #     num = sum([v for k, v in neighs.items() if k in user_items])
    #     den = sum(np.power(list(neighs.values()), 1))
    #     return num/den if den != 0 else 0

    def get_model_state(self):
        saving_dict = {}
        saving_dict['_preds'] = self._preds
        saving_dict['_similarity'] = self._similarity
        saving_dict['_num_neighbors'] = self._num_neighbors
        saving_dict['_implicit'] = self._implicit
        return saving_dict

    def set_model_state(self, saving_dict):
        self._preds = saving_dict['_preds']
        self._similarity = saving_dict['_similarity']
        self._num_neighbors = saving_dict['_num_neighbors']
        self._implicit = saving_dict['_implicit']

    def load_weights(self, path):
        with open(path, "rb") as f:
            self.set_model_state(pickle.load(f))

    def save_weights(self, path):
        with open(path, "wb") as f:
            pickle.dump(self.get_model_state(), f)


def compute_similarity(candidates_matrix_chunk, item_user_matrix, idx):
    n_items = candidates_matrix_chunk.shape[0]
    data, rows_indices = [], []
    for i in range(n_items):
        # Get the indices of the candidates for the i-th item
        candidate_indices = candidates_matrix_chunk[i]

        # Extract the relevant vectors from URM for these candidates
        URM_candidates = item_user_matrix[candidate_indices, :]

        # Compute cosine similarity between item i and its candidates
        item_vector = item_user_matrix[i, :]
        sim_scores = cosine_similarity(item_vector, URM_candidates)
        data.extend(sim_scores.squeeze())
        rows_indices.extend(candidate_indices)
    return data, rows_indices, idx
