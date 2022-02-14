import os

import pandas as pd
from pandas import DataFrame as df
import json


##INPUT_DIR = input("input directory: ")
INPUT_DIR = "C:\\Users\\Al Zoghby\\PycharmProjects\\Image-Data-Annotation\\assays\\CTCF-AID"
with open(os.path.join(INPUT_DIR, 'assay_config.json'), mode="r") as config_file:
    config = json.load(config_file)

COLUMN_NAMES = config["column_names"]


files_list = os.listdir(INPUT_DIR)
table = df(columns=COLUMN_NAMES)

for file_name in files_list:
    if file_name.endswith(".tif"):
        line = [[file_name]+file_name.split(sep="_")]
        line = df(line, columns=COLUMN_NAMES)
        table = pd.concat([table, line], ignore_index=True)
print(table)

probe_table = pd.read_csv(".\meta_data\probe_table.csv")
print(probe_table)

probe_table_ch0 = probe_table.rename(columns=lambda x: x+"_ch0")
probe_table_ch1 = probe_table.rename(columns=lambda x: x+"_ch1")
table = pd.merge(table, probe_table_ch0, left_on=table["probe_ch0"].apply(lambda x: x.split(sep="-")[0]), right_on="probe ID_ch0")
table = pd.merge(table, probe_table_ch1, left_on=table["probe_ch1"].apply(lambda x: x.split(sep="-")[0]), right_on="probe ID_ch1")

print(table)

