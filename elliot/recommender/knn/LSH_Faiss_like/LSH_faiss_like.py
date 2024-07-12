import os

import numpy as np
from multiprocessing import Pool, cpu_count
from heapq import heappush, heappop
from ..LSH_v4.utils import create_sparse_matrix
import time


class IndexLSH:
    def __init__(self, d, nbits, initialization="uniform"):
        self.d = d
        self.nbits = nbits
        self.codes = None
        self.projection_matrix = self._initialize_projection_matrix(initialization=initialization)

    def apply_preprocess(self, x):
        return x.dot(self.projection_matrix)

    def sa_encode(self, x):
        x = self.apply_preprocess(x)
        return (x >= 0).astype(np.uint8)

    def add(self, x):
        if self.codes is None:
            self.codes = self.sa_encode(x)
        else:
            self.codes = np.vstack([self.codes, self.sa_encode(x)])

    def hamming_distance(self, a, b):
        return np.count_nonzero(a != b, axis=1)

    def search_single_query(self, query, k):
        # Usando argpartition limitiamo la complessità di questa funzione
        distances = self.hamming_distance(self.codes, query)
        nearest_indices = distances.argpartition(k)[:k]  # Get the indices of the k smallest distances
        sorted_indices = nearest_indices[np.argsort(distances[nearest_indices])]  # Sort the k smallest distances
        return distances[sorted_indices], sorted_indices

    def search_batch(self, batch_queries, k):
        results = []
        for query in batch_queries:
            distances, labels = self.search_single_query(query, k)
            results.append((distances, labels))
        return results

    def search_2(self, x, k):
        prima = time.time()
        query_codes = self.sa_encode(x)
        print("TEMPO PER INDICIZZARE LE QUERY", time.time() - prima)
        num_workers = os.cpu_count()
        num_queries = len(query_codes)
        chunk_size = max(1, num_queries // num_workers)
        distances = np.zeros((num_queries, k))
        labels = np.zeros((num_queries, k), dtype=int)

        # Split queries into chunks to be processed by available CPU cores
        chunks = [query_codes[i:i + chunk_size] for i in range(0, num_queries, chunk_size)]

        with Pool(cpu_count() // 2) as pool:
            results = pool.starmap(self.search_batch, [(chunk, k) for chunk in chunks])

        results_flat = [item for sublist in results for item in sublist]

        prima = time.time()
        for i, (dist, lbl) in enumerate(results_flat):
            distances[i] = dist
            labels[i] = lbl
        print("TEMPO ITERARE SUI RISULTATI", time.time() - prima)

        return distances, labels

    def _initialize_projection_matrix(self, initialization="uniform"):
        """
        Useful to inizialize a matrix for projecting our dense vectors in binary ones
        :return:
        """
        print(initialization)
        if initialization == "gaussian":
            return np.random.normal(0, 1, (self.d, self.nbits))
        else:
            return np.random.rand(self.d, self.nbits) - .5


# Example usage
if __name__ == "__main__":
    seed = 42
    nbits = 4  # number of hyperplanes in a three --> Decrease the number of false positives
    m = 6400  # number of users
    n = 20000  # number of items
    sparsity = 0.99
    neighbours = 200
    user_item_matrix_dummy = create_sparse_matrix(m, n, sparsity=sparsity, seed=seed)
    item_user_matrix_dummy = user_item_matrix_dummy.T.tocsr()
    sparsity = 0.9
    neighbours = 200
    lsh_index = IndexLSH(d=m, nbits=nbits)
    prima = time.time()
    lsh_index.add(item_user_matrix_dummy)
    print(time.time() - prima, "tempo per indicizzare gli elementi")
    prima = time.time()
    distances, labels = lsh_index.search_2(item_user_matrix_dummy, k=neighbours)
    dopo = time.time()
    print(dopo - prima, "tempo per restituire i candidati")
    print(labels)
    print(distances)
