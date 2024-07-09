"""
Script made to compare a baseline with the corresponding LSH version
"""
from elliot.utils.lsh_utils import find_best_experiment_static
import os
import datetime

model = "ItemKNN"
baseline_path = f"results_lsh/comparisons/movielens_1m/UserKNN/UserKNN_baseline.tsv"
lsh_path = f"results_lsh/comparisons/movielens_1m_faissCopy/UserKNN/UserKNN_rp_faiss.tsv"
print("Last Modification Date of the Baseline: ", datetime.datetime.fromtimestamp(os.path.getmtime(baseline_path)))
print("Last Modification Date of the LSH Approach: ", datetime.datetime.fromtimestamp(os.path.getmtime(lsh_path)))
find_best_experiment_static(baseline_path, lsh_path)
