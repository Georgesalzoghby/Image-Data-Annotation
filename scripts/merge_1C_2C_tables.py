import pandas as pd
import numpy as np
import os

# metadata_1_path = "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC/ESC_1C/Experiment_A_assays.csv"
# metadata_2_path = "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC/ESC_2C/Experiment_A_assays.csv"
# metadata_path = "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC/Experiment_A_assays.csv"
# metadata_1_path = "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC_TSA/ESC_TSA_1C/Experiment_I_assays.csv"
# metadata_2_path = "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC_TSA/ESC_TSA_2C/Experiment_I_assays.csv"
# metadata_path = "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC_TSA/Experiment_I_assays.csv"
# metadata_1_path = "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC_TSA-CTL/ESC_TSA-CTL_1C/Experiment_H_assays.csv"
# metadata_2_path = "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC_TSA-CTL/ESC_TSA-CTL_2C/Experiment_H_assays.csv"
# metadata_path = "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC_TSA-CTL/Experiment_H_assays.csv"
metadata_1_path = "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/NPC/NPC_1C/Experiment_B_assays.csv"
metadata_2_path = "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/NPC/NPC_2C/Experiment_B_assays.csv"
metadata_path = "/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/NPC/Experiment_B_assays.csv"

metadata_df_1 = pd.read_csv(metadata_1_path,
                            header=1)
metadata_df_2 = pd.read_csv(metadata_2_path,
                            header=1)

metadata_df = pd.concat([metadata_df_1, metadata_df_2], axis=0, ignore_index=False)

csv_header = "# header "
for dt in list(metadata_df.dtypes):
    if dt == np.int64:
        csv_header = csv_header + 'l,'
    elif dt == np.float64:
        csv_header = csv_header + 'd,'
    else:
        csv_header = csv_header + 's,'

with open(metadata_path, mode='w') as csv_file:
    csv_file.write(csv_header[:-1])
    csv_file.write("\n")
    csv_file.write(metadata_df.to_csv(index=False, line_terminator='\n'))

# metadata_df.to_csv(metadata_path,
#                    # header=False,
#                    index=False)

# analysis_df = pd.read_csv("/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC_TSA/analysis_df.csv")
#
# merge_df = pd.merge(metadata_df, analysis_df, on="Image Name")
#
# metadata_df = pd.read_csv("/home/julio/Documents/data-annotation/Image-Data-Annotation/assays/ESC_TSA/Experiment_I_assays.csv"), header=1)  # TODO:
