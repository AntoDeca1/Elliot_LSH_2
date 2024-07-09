"""
This script has the scope of comparing two implementations
N.B: The two implementations to be comparable need to have the same experiments settings
"""
import os
import datetime
from elliot.utils.lsh_utils import compare_experiments

first_experiments_path = "results_lsh/comparisons/amazon_music_large/UserKNN/UserKNN_rp_custom.tsv"
second_experiments_path = "results_lsh/comparisons/amazon_music_large_faiss_copy/UserKNN/UserKNN_rp_custommp.tsv"

print("Last Modification Date of the First Experiment: ",
      datetime.datetime.fromtimestamp(os.path.getmtime(first_experiments_path)))
print("Last Modification Date of the Second Experiment: ",
      datetime.datetime.fromtimestamp(os.path.getmtime(second_experiments_path)))
compare_experiments(first_experiments_path, second_experiments_path)
