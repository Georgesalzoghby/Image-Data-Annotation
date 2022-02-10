import os

from pandas import DataFrame as df

INPUT_DIR = input("input directory: ")


files_list = os.listdir(INPUT_DIR)
table = df()

for file_name in files_list:
    if file_name.endswith(".tif"):
        print(file_name.split(sep="_"))
