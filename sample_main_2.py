import zipfile
import io
import requests
import os
from elliot.run import run_experiment

print("Done! We are now starting the Elliot's experiment.")
run_experiment("config_files/epinions_baselines.yml")
