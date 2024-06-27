from elliot.utils.lsh_utils import average_column_from_subdirectories

main_path = "results_lsh/experiments/movielens_1m"
model = "UserKNN"
similarity = "rp_faiss"  # rp_custom,rp_hashtables
avg_df = average_column_from_subdirectories(main_path, model, similarity)
