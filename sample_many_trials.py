import zipfile
import io
import requests
import os
from elliot.run import run_experiment

# AGGIUNTO LA PROTEZIONE DEL MAIN PER MULTIPROCESSING
if __name__ == '__main__':
    trials = 50
    print("Done! We are now starting the Elliot's experiment.")
    for _ in range(trials):
        print(f"-------Starting trials number {_}------- ")
        run_experiment("config_files/paper_reproducible_config.yml")
        print(f"------Ended trials number {_}-------- ")
