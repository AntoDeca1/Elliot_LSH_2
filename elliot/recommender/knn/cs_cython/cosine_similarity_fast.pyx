# cosine_similarity_fast.pyx
from libc.math cimport sqrt
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
import time

def compute_candidates_cosine_similarity_np(np.ndarray[np.float64_t, ndim=2] user_item_matrix,
                                            np.ndarray[np.int32_t, ndim=2] candidate_matrix):
    cdef:
        int num_users = user_item_matrix.shape[0]
        int num_features = user_item_matrix.shape[1]
        int num_candidates = candidate_matrix.shape[1]
        int user_idx, candidate_idx, feature_idx
        double sum_sq_u, sum_sq_c, dot_product
        np.ndarray[np.float64_t, ndim=2] similarities = np.zeros((num_users, num_candidates), dtype=np.float64)
        np.ndarray[np.float64_t, ndim=3] candidates_users = np.zeros((num_users, num_candidates, num_features), dtype=np.float64)

    # Populate candidates_users by checking indices
    for user_idx in range(num_users):
        for candidate_idx in range(num_candidates):
            if candidate_matrix[user_idx, candidate_idx] < num_users:  # Ensure index is within bounds
                for feature_idx in range(num_features):
                    candidates_users[user_idx, candidate_idx, feature_idx] = user_item_matrix[candidate_matrix[user_idx, candidate_idx], feature_idx]

    # Start timing
    start_time = time.time()

    # Compute cosine similarity
    for user_idx in range(num_users):
        for candidate_idx in range(num_candidates):
            dot_product = 0.0
            sum_sq_u = 0.0
            sum_sq_c = 0.0
            for feature_idx in range(num_features):
                dot_product += candidates_users[user_idx, candidate_idx, feature_idx] * user_item_matrix[user_idx, feature_idx]
                sum_sq_u += user_item_matrix[user_idx, feature_idx] ** 2
                sum_sq_c += candidates_users[user_idx, candidate_idx, feature_idx] ** 2

            if sum_sq_u > 0 and sum_sq_c > 0:
                similarities[user_idx, candidate_idx] = dot_product / (sqrt(sum_sq_u) * sqrt(sum_sq_c))

    # Stop timing and print
    print("Time needed:", time.time() - start_time)

    return similarities
