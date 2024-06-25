"""
Script made to compare a baseline with the corresponding LSH version
"""
from elliot.utils.lsh_utils import find_best_experiment_static
import os
import datetime

baseline_path = "results_lsh/comparisons/amazon_music_large_3/UserKNN/UserKNN_baseline.tsv"
lsh_path = "results_lsh/comparisons/amazon_music_large_3/UserKNN/UserKNN_rp_hashtables.tsv"
print("Last Modification Date of the Baseline: ", datetime.datetime.fromtimestamp(os.path.getmtime(baseline_path)))
print("Last Modification Date of the LSH Approach: ", datetime.datetime.fromtimestamp(os.path.getmtime(lsh_path)))
find_best_experiment_static(baseline_path, lsh_path)
