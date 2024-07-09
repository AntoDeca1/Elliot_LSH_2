import zipfile
import io
import requests
import os
from elliot.run import run_experiment

if __name__ == "__main__":
    # dataset=[movielens,epinions,ADM,ADM_large]
    dataset = "ADM_large"
    print("Done! We are now starting the Elliot's experiment.")
    run_experiment(f"prefilter_{dataset}.yml")
