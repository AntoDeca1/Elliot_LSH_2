import pickle

import numpy as np
import scipy.sparse
import sklearn.preprocessing
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, haversine_distances, chi2_kernel, \
    manhattan_distances
from sklearn.metrics import pairwise_distances
import time
from ..LSH_HashTables.LSH import LSHRp as LSHHashTables
from ..LSH_v4.LSH_v4 import RandomProjections as CustomLSH
from ..LSH_faiss.LSH_faiss import RandomProjections as FaissLSH
from ..LSH_v3.LSH_v3 import RandomProjections as LSH_noHamming
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
from tqdm import tqdm
import gc
import torch
import pyximport
from opt_einsum import contract


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

        # CODICE AGGIUNTO PER LSH
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

        print(f"Similarity used : {self._similarity} with {self._num_neighbors}")

        self._similarity_matrix = np.empty((len(self._users), len(self._users)))

        ##############
        data, rows_indices, cols_indptr = [], [], []

        if self._similarity in ["rp_faiss", "rp_custom"]:
            data, rows_indices, cols_indptr = self.lsh_similarity(self._URM)
        else:
            self.process_similarity(self._similarity)

            column_row_index = np.arange(len(self._users), dtype=np.int32)
            for user_idx in range(len(self._users)):
                cols_indptr.append(len(data))
                column_data = self._similarity_matrix[user_idx, :]

                non_zero_data = column_data != 0

                idx_sorted = np.argsort(column_data[non_zero_data])  # sort by column
                top_k_idx = idx_sorted[-self._num_neighbors:]

                data.extend(column_data[non_zero_data][top_k_idx])
                rows_indices.extend(column_row_index[non_zero_data][top_k_idx])

            cols_indptr.append(len(data))

        W_sparse = sparse.csc_matrix((data, rows_indices, cols_indptr),
                                     shape=(len(self._users), len(self._users)), dtype=np.float32).tocsr()
        self._preds = W_sparse.dot(self._URM).toarray()

        del self._similarity_matrix

    def process_similarity(self, similarity):
        if similarity == "cosine":
            self._similarity_matrix = cosine_similarity(self._URM)
        elif similarity == "dot":
            self._similarity_matrix = (self._URM @ self._URM.T).toarray()
        elif similarity == "euclidean":
            self._similarity_matrix = (1 / (1 + euclidean_distances(self._URM)))
        elif similarity == "manhattan":
            self._similarity_matrix = (1 / (1 + manhattan_distances(self._URM)))
        elif similarity == "haversine":
            self._similarity_matrix = (1 / (1 + haversine_distances(self._URM)))
        elif similarity == "chi2":
            self._similarity_matrix = (1 / (1 + chi2_kernel(self._URM)))
        elif similarity == "baseline":
            self._similarity_matrix = self.baseline(self._URM)
        elif similarity == "rp_hashtables":
            print(f"{self._similarity} similarity with nbits: {self._nbits} and ntables: {self._ntables} ")
            lsh_index = LSH_noHamming(d=len(self._items), nbits=self._nbits, l=self._ntables)
            prima = time.time()
            lsh_index.add(self._URM)
            indexing_time = time.time() - prima
            print(indexing_time, "Time to index the vectors")
            self._lsh_times_obj["indexing_time"] = indexing_time
            prima = time.time()
            candidates_matrix = lsh_index.search_2()
            candidates_retrieval_time = time.time() - prima
            print(candidates_retrieval_time, "Time to pull out the candidates")
            prima = time.time()
            self._similarity_matrix = self.compute_candidates_cosine_similarity(user_item_matrix=self._URM,
                                                                                candidate_matrix=candidates_matrix)
            similarity_matrix_time = time.time() - prima
            print(similarity_matrix_time, "Time for calculating the similarity matrix")
            self._lsh_times_obj["similarity_matrix_time"] = similarity_matrix_time

            self._lsh_times_obj["candidates_retrieval_time"] = candidates_retrieval_time
        elif similarity in ['cityblock', 'l1', 'l2']:
            self._similarity_matrix = (1 / (1 + pairwise_distances(self._URM, metric=similarity)))
        elif similarity in ['braycurtis', 'canberra', 'chebyshev', 'correlation', 'dice', 'hamming', 'jaccard',
                            'kulsinski', 'mahalanobis', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean',
                            'sokalmichener', 'sokalsneath', 'sqeuclidean', 'yule']:
            self._similarity_matrix = (1 / (1 + pairwise_distances(self._URM.toarray(), metric=similarity)))
        else:
            raise ValueError("Compute Similarity: value for parameter 'similarity' not recognized."
                             f"\nAllowed values are: {self.supported_similarities}, {self.supported_dissimilarities}."
                             f"\nPassed value was {similarity}\nTry with implementation: aiolli")

    def lsh_similarity(self, user_item_matrix):
        """
        1) Instantiate the Index
        2) Project vector in a lower dimensional space
        3) Pick the candidates
        4) Compute the cosine similarity starting from the candidates taken from LSH
        :param item_user_matrix:
        :return:
        """
        if self._similarity == "rp_faiss":
            print(f"{self._similarity} similarity with nbits: {self._nbits} and ntables: {self._ntables} ")
            rp = FaissLSH(d=len(self._items), nbits=self._nbits)
        else:
            print(f"{self._similarity} similarity with nbits: {self._nbits} and ntables: {self._ntables} ")
            rp = CustomLSH(d=len(self._items), nbits=self._nbits, l=self._ntables)

        prima = time.time()
        rp.add(user_item_matrix)
        indexing_time = time.time() - prima
        self._lsh_times_obj["indexing_time"] = indexing_time

        print(indexing_time, "Time to index the vectors")

        prima = time.time()
        if isinstance(rp, FaissLSH):
            candidates_matrix = rp.search_2(user_item_matrix, k=self._num_neighbors)
        else:
            candidates_matrix = rp.search_2(k=self._num_neighbors)
        candidates_retrieval_time = time.time() - prima
        print(candidates_retrieval_time, "Time to pull out the candidates")
        self._lsh_times_obj["candidates_retrieval_time"] = candidates_retrieval_time
        prima = time.time()
        del rp
        print("Time taken to remove the object from memory", time.time() - prima)
        # gc.collect()

        prima = time.time()
        data, rows_indices, cols_indptr = self.compute_candidates_cosine_similarity_fast(user_item_matrix,
                                                                                         candidates_matrix)

        similarity_matrix_time = time.time() - prima
        print(similarity_matrix_time, "Time for calculating the similarity matrix")
        self._lsh_times_obj["similarity_matrix_time"] = similarity_matrix_time
        return data, rows_indices, cols_indptr

    def compute_candidates_cosine_similarity_np(self, user_item_matrix: np.array,
                                                candidate_matrix: np.array):
        """
         In the case of Faiss or CustomLSH(my implementation of Faiss) we already retrieve the k candidates
         It makes no sense to sort them relatevely to the cosine similarity and so we skip this step and we output directly the components of the final sparse matrix
         :param user_item_matrix:
         :param item_user_matrix:
         :param candidate_matrix:
         :return:
        """
        dense_user_item_matrix = user_item_matrix.A
        candidates_users = dense_user_item_matrix[candidate_matrix]
        prima = time.time()
        similarities = np.einsum("ijk,ik->ij", candidates_users, dense_user_item_matrix, optimize=True) / (np.sqrt(
            np.sum(candidates_users ** 2, axis=2)) * np.sqrt(
            np.sum(dense_user_item_matrix ** 2, axis=1)[:, np.newaxis]))
        print(time.time() - prima, "Tempo necessario")
        return similarities

    def baseline(self, user_item_matrix):
        """
        In the case of Faiss or CustomLSH(my implementation of Faiss) we already retrieve the k candidates
        It makes no sense to sort them relatevely to the cosine similarity and so we skip this step and we output directly the components of the final sparse matrix
        :param item_user_matrix:
        :param candidate_matrix:
        :return:
        """
        n_users = user_item_matrix.shape[0]
        similarity_matrix = np.empty((n_users, n_users))
        for i in range(n_users):
            # Compute cosine similarity between item i and its candidates
            user_vector = user_item_matrix[i]
            sim_scores = cosine_similarity(user_vector, user_item_matrix)
            similarity_matrix[i] = sim_scores
        return similarity_matrix

    def compute_candidates_cosine_similarity_fast(self, user_item_matrix, candidate_matrix):
        """
        In the case of Faiss or CustomLSH(my implementation of Faiss) we already retrieve the k candidates
        It makes no sense to sort them relatevely to the cosine similarity and so we skip this step and we output directly the components of the final sparse matrix
        :param item_user_matrix:
        :param candidate_matrix:
        :return:
        """
        n_users = candidate_matrix.shape[0]
        data, rows_indices, cols_indptr = [], [], []
        for i in range(n_users):
            cols_indptr.append(len(data))
            # Get the indices of the candidates for the i-th item
            candidate_indices = candidate_matrix[i]

            # Extract the relevant vectors from URM for these candidates
            URM_candidates = user_item_matrix[candidate_indices, :]
            # Compute cosine similarity between item i and its candidates
            user_vector = user_item_matrix[i]
            sim_scores = cosine_similarity(user_vector, URM_candidates)
            data.extend(sim_scores.squeeze())
            rows_indices.extend(candidate_indices)
        cols_indptr.append(len(data))

        return data, rows_indices, cols_indptr

    def compute_candidates_cosine_similarity_mp(self, item_user_matrix, candidate_matrix):
        """
        For each item(in case of ItemKNN) pick its candidates and populate the corrisponding row in the similarity matrix
        :param item_user_matrix:
        :param candidate_matrix:
        :return:
        """
        n_items = candidate_matrix.shape[0]
        # MULTIPROCESSING HERE POSSIBLY
        n_processes = mp.cpu_count() // 2
        similarity_matrix = np.zeros((n_items, n_items))

        with ProcessPoolExecutor(max_workers=n_processes) as executor:
            chunk_size = n_items // n_processes
            tasks = []
            for idx in range(0, item_user_matrix.shape[0], chunk_size):
                data_chunk = [item_user_matrix[idx * chunk_size:(idx + 1) * chunk_size],
                              item_user_matrix[
                                  candidate_matrix[idx * chunk_size:(idx + 1) * chunk_size].nonzero()[1].reshape(-1,
                                                                                                                 self._num_neighbors).flatten()],
                              [candidate_matrix[idx * chunk_size:(idx + 1) * chunk_size].nonzero()][0][1].reshape(-1,
                                                                                                                  self._num_neighbors),
                              idx, n_items]
                tasks.append(executor.submit(compute_similarity, data_chunk))
            for future in tqdm(as_completed(tasks), total=len(tasks)):
                similarity_chunk, idx = future.result()
                similarity_matrix[idx * chunk_size:(idx + 1) * chunk_size] = similarity_chunk
            self._similarity_matrix = similarity_matrix

    def compute_candidates_cosine_similarity(self, user_item_matrix, candidate_matrix):
        n_items = candidate_matrix.shape[0]
        similarity_matrix = np.empty((len(self._users), len(self._users)))
        # MULTIPROCESSING HERE
        for i in range(n_items):
            # Get the indices of the candidates for the i-th item
            candidate_indices = candidate_matrix.getrow(i).nonzero()[1]

            # Extract the relevant vectors from URM for these candidates
            URM_candidates = user_item_matrix[candidate_indices, :]

            # Compute cosine similarity between item i and its candidates
            item_vector = user_item_matrix[i, :]
            sim_scores = cosine_similarity(item_vector, URM_candidates)

            # Store the results
            similarity_matrix[i, candidate_indices] = sim_scores
        return similarity_matrix

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

    # def get_user_recs(self, u, mask, k):
    #     user_items = self._ratings[u].keys()
    #     user_mask = mask[self._data.public_users[u]]
    #     predictions = {i: self.score_item(self.get_user_neighbors(u), user_items) for i in self._data.items if
    #                    user_mask[self._data.public_items[i]]}
    #
    #     # user_items = self._ratings[u].keys()
    #     # predictions = {i: self.score_item(self.get_user_neighbors(u), self._item_ratings[i].keys())
    #     #                for i in self._data.items if i not in user_items}
    #
    #     indices, values = zip(*predictions.items())
    #     indices = np.array(indices)
    #     values = np.array(values)
    #     local_k = min(k, len(values))
    #     partially_ordered_preds_indices = np.argpartition(values, -local_k)[-local_k:]
    #     real_values = values[partially_ordered_preds_indices]
    #     real_indices = indices[partially_ordered_preds_indices]
    #     local_top_k = real_values.argsort()[::-1]
    #     return [(real_indices[item], real_values[item]) for item in local_top_k]

    # @staticmethod
    # def score_item(neighs, user_neighs_items):
    #     num = sum([v for k, v in neighs.items() if k in user_neighs_items])
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


def compute_similarity(data_chunk):
    try:
        item_user_matrix_chunk, candidates_vectors_extended, candidate_indices, idx, tot_items = data_chunk
        n_items = item_user_matrix_chunk.shape[0]
        n_neighbors = candidate_indices.shape[1]
        similarity_matrix = np.zeros((n_items, tot_items))
        for i in range(n_items):
            # Compute cosine similarity between item i and its candidates
            candidates_vector = candidates_vectors_extended[i * n_neighbors: (i + 1) * n_neighbors]
            item_vector = item_user_matrix_chunk[i, :]
            sim_scores = cosine_similarity(item_vector, candidates_vector)

            # Store the results
            similarity_matrix[i, candidate_indices] = sim_scores
        return similarity_matrix, idx
    except Exception as e:
        print("Error in compute_similarity:", e)
        return None
