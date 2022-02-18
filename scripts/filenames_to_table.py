import os

import pandas as pd
from pandas import DataFrame as df
import json


INPUT_DIR = input("input directory: ")
#INPUT_DIR = "C:\\Users\\Al Zoghby\\PycharmProjects\\Image-Data-Annotation\\assays\\CTCF-AID"
with open(os.path.join(INPUT_DIR, 'assay_config.json'), mode="r") as config_file:
    config = json.load(config_file)

COLUMN_NAMES = config["column_names"]
MERGES = config["merges"]


files_list = os.listdir(INPUT_DIR)
table = df(columns=COLUMN_NAMES)

for file_name in files_list:
    if file_name.endswith(".tif"):
        line = [[file_name]+file_name.split(sep="_")]
        line = df(line, columns=COLUMN_NAMES)
        table = pd.concat([table, line], ignore_index=True)

for col_name, table_file in MERGES.items():
    merge_table = pd.read_csv(os.path.join(".", "meta_data", table_file))
    table = pd.merge(table, merge_table, left_on=col_name, right_on="on", how='left', suffixes=("_ch0" , "_ch1"))
table = table.drop(columns = ["on_ch0", "on_ch1"])

print(table)

