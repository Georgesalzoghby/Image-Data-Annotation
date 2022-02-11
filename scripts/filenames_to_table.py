import os

import pandas as pd
from pandas import DataFrame as df
import json


INPUT_DIR = input("input directory: ")
with open(os.path.join(INPUT_DIR, 'assay_config.json'), mode="r") as config_file:
    config = json.load(config_file)

COLUMN_NAMES = config["column_names"]


files_list = os.listdir(INPUT_DIR)
table = df(columns=COLUMN_NAMES)

for file_name in files_list:
    if file_name.endswith(".tif"):

        line = df([file_name.split(sep="_")], columns=COLUMN_NAMES)
        table = pd.concat([table, line], ignore_index=True)

print(table)
probes = pd.read_csv("probes.CSV")

print(probes)
table = pd.merge(table, probes, on="probe ID", how="left")
print(table)



