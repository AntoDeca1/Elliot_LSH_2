from elliot.utils.lsh_utils import create_subplots
import datetime
import os

data_path = "results_lsh/comparisons/movielens_1m/UserKNN/UserKNN_rp_faiss.tsv"
print("Last Modification Date of the tsv file: ",
      datetime.datetime.fromtimestamp(os.path.getmtime(data_path)))
create_subplots(data_path)
