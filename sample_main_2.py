import zipfile
import io
import requests
import os
from elliot.run import run_experiment

# AGGIUNTO LA PROTEZIONE DEL MAIN PER MULTIPROCESSING
if __name__ == '__main__':
    # Directory and file paths
    data_dir = "data/movielens_1m"
    dataset_file = os.path.join(data_dir, "dataset.tsv")

    # URL of the dataset
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    print(f"Checking if dataset exists at: {dataset_file}")

    # Check if the dataset file already exists
    if not os.path.exists(dataset_file):
        print(f"Getting Movielens 1Million from: {url} ..")
        response = requests.get(url)
        ml_1m_ratings = []

        print("Extracting ratings.dat ..")
        with zipfile.ZipFile(io.BytesIO(response.content)) as zip_ref:
            for line in zip_ref.open("ml-1m/ratings.dat"):
                ml_1m_ratings.append(str(line, "utf-8").replace("::", "\t"))

        print(f"Printing ratings.tsv to {data_dir} ..")
        os.makedirs(data_dir, exist_ok=True)
        with open(dataset_file, "w") as f:
            f.writelines(ml_1m_ratings)

        print("Download and extraction complete.")
    else:
        print("Dataset already exists. Skipping download and extraction.")

    print("Done! We are now starting the Elliot's experiment.")
    run_experiment("config_files/amazon_large.yml")
