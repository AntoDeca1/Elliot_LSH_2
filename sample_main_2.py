import zipfile
import io
import requests
import os
from elliot.run import run_experiment

if __name__ == "__main__":
    print("Done! We are now starting the Elliot's experiment.")
    run_experiment("config_files/paper_reproducible_config.yml")
