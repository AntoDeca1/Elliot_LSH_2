"""
Script made to compare a baseline with the corresponding LSH version
"""
from elliot.utils.lsh_utils import find_bext_experiment_interactive, find_best_experiment_static

baseline_path = "../elliot/results_lsh/comparisons/amazon_music_large/ItemKNN/ItemKNN_baseline.tsv"
lsh_path = "../elliot/results_lsh/comparisons/amazon_music_large/ItemKNN/ItemKNN_rp_hashtables.tsv"

find_best_experiment_static(baseline_path, lsh_path)
