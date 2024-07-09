import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Run split user/item groups with pareto.")
parser.add_argument('--dataset', type=str, default='amazon_music_large')
args = parser.parse_args()

dataset = args.dataset
if dataset == 'movielens_1m':
    df = pd.read_csv(f'./data/{dataset}/filtered_data/0/train.tsv', sep='\t', header=None,
                     names=['user_id', 'item_id', 'rating', 'timestamp'])
elif dataset in ['amazon_music', 'epinions', "amazon_music_large"]:
    df = pd.read_csv(f'./data/{dataset}/filtered_data/0/train.tsv', sep='\t', header=None,
                     names=['user_id', 'item_id', 'rating', 'timestamp'])
    df = df.drop('timestamp', axis=1)
print("data successfully loaded")

# Compute the number of times a user(item) appear in a transaction
user_count = df['user_id'].value_counts()
item_count = df['item_id'].value_counts()
print("Sorting the values...")

# Order the users(items) relatively to their counts
user_frequencies = pd.DataFrame(
    data=np.concatenate([user_count.index.values.reshape(-1, 1), user_count.values.reshape(-1, 1)],
                        axis=1)).sort_values(by=1, ascending=True)
item_frequencies = pd.DataFrame(
    data=np.concatenate([item_count.index.values.reshape(-1, 1), item_count.values.reshape(-1, 1)],
                        axis=1)).sort_values(by=1, ascending=True)
print("Computing the quantiles for the distributions...")

user_25 = np.quantile(user_frequencies[1], 0.25)
user_50 = np.quantile(user_frequencies[1], 0.50)
user_75 = np.quantile(user_frequencies[1], 0.75)
item_25 = np.quantile(item_frequencies[1], 0.25)
item_50 = np.quantile(item_frequencies[1], 0.50)
item_75 = np.quantile(item_frequencies[1], 0.75)

user_bins = [0, user_25, user_50, user_75, np.inf]
user_bins = list(set(user_bins))
user_bins.sort()

user_labels = [i for i in range(len(user_bins) - 1)]

item_bins = [0, item_25, item_50, item_75, np.inf]
item_bins = list(set(item_bins))
item_bins.sort()
item_labels = [i for i in range(len(item_bins) - 1)]

print("Assigning the labels to each bin")
user_frequencies['group'] = pd.cut(user_frequencies[1], bins=user_bins, labels=user_labels, right=True)
item_frequencies['group'] = pd.cut(item_frequencies[1], bins=item_bins, labels=item_labels, right=True)

# create a column for when the users/items are not split
user_frequencies['no_group'] = np.full(shape=user_frequencies.shape[0], fill_value=0)
item_frequencies['no_group'] = np.full(shape=item_frequencies.shape[0], fill_value=0)

print("Saving the results...")
# save the grouping of ID-group
user_frequencies[[0, 'group']].to_csv(f'data/{dataset}/user_groups_4.tsv', sep='\t', header=None, index=None)
item_frequencies[[0, 'group']].to_csv(f'data/{dataset}/item_groups_4.tsv', sep='\t', header=None, index=None)

# save the dataset without grouping
user_frequencies[[0, 'no_group']].to_csv(f'data/{dataset}/user_no_groups.tsv', sep='\t', header=None, index=None)
item_frequencies[[0, 'no_group']].to_csv(f'data/{dataset}/item_no_groups.tsv', sep='\t', header=None, index=None)
