import os
import numpy as np
import pandas as pd
from pandas import DataFrame as df
import json

#INPUT_DIR = input("input directory: ")
INPUT_DIR = "C:\\Users\\Al Zoghby\\PycharmProjects\\Image-Data-Annotation\\assays\\CTCF-AID"
with open(os.path.join(INPUT_DIR, 'assay_config.json'), mode="r") as config_file:
    config = json.load(config_file)

COLUMN_NAMES = config["column_names"]
MERGES = config["merges"]


files_list = os.listdir(INPUT_DIR)
table = df(columns=COLUMN_NAMES)

for file_name in files_list:
    # if file_name.endswith(".ome-tif"):
    if file_name.endswith(".tif"):
        line = [[file_name]+file_name.split(sep="_")]
        # line[0][-1] = line[0][-1].removesuffix(".ome-tif")
        line[0][-1] = line[0][-1].removesuffix(".tif")
        line = df(line, columns=COLUMN_NAMES)
        table = pd.concat([table, line], ignore_index=True)

for col_name, table_file in MERGES.items():
    merge_table = pd.read_csv(os.path.join(".", "meta_data", table_file))
    table = pd.merge(table, merge_table, left_on=col_name, right_on="on", how='left', suffixes=(" Ch0", " Ch1"))
table = table.drop(columns=["on Ch0", "on Ch1"])

if 'NPC' in INPUT_DIR:
    table = table.drop(columns=["Cluster ESC Ch0", "Cluster ESC Ch1"])
else:
    table = table.drop(columns=["Cluster NPC Ch0", "Cluster NPC Ch1"])

csv_header = "# header "
for dt in list(table.dtypes):
    if dt == np.int64:
        csv_header = csv_header + 'l,'
    elif dt == np.float64:
        csv_header = csv_header + 'd,'
    else:
        csv_header = csv_header + 's,'
print(csv_header[:-1])

with open(os.path.join(INPUT_DIR, 'table.csv'), mode='w') as csv_file:
    csv_file.write(csv_header[:-1])
    csv_file.write("\n")
    csv_file.write(table.to_csv(index=False, line_terminator='\n'))

# print(table)
table.to_csv(os.path.join(INPUT_DIR, 'table.csv'), index=False)