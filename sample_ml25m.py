import zipfile
import io
import requests
import os
import csv  # Importing CSV module to handle CSV files
from elliot.run import run_experiment

# Directory and file paths
data_dir = "data/movielens_25m"
dataset_file = os.path.join(data_dir, "dataset.csv")  # Changed to .csv

# URL of the dataset
url = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
print(f"Checking if dataset exists at: {dataset_file}")

# Check if the dataset file already exists
if not os.path.exists(dataset_file):
    print(f"Getting Movielens 25M from: {url} ..")
    response = requests.get(url)
    ml_25m_ratings = []

    print("Extracting ratings.csv ..")
    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
        # Using csv reader to handle the CSV file
        with zip_ref.open("ml-25m/ratings.csv") as file:
            reader = csv.reader(io.TextIOWrapper(file))
            next(reader)  # Skip the header row
            for row in reader:
                ml_25m_ratings.append("\t".join(row))  # Join columns with tab

    print(f"Printing ratings.tsv to {data_dir} ..")
    os.makedirs(data_dir, exist_ok=True)
    with open(dataset_file, "w") as f:
        f.writelines("\n".join(ml_25m_ratings))  # Join rows with newlines

    print("Download and extraction complete.")
else:
    print("Dataset already exists. Skipping download and extraction.")

print("Done! We are now starting the Elliot's experiment.")
run_experiment("config_files/epinions.yml")
