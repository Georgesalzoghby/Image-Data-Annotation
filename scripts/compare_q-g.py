"""This script is comparing the data provided by the script of Julio and Georges with the one of Quentin"""


import pandas as pd

quentin_df = pd.read_csv('./Quentin_raw_meassurements/quentin_measurement_file.csv')
georges_df = pd.read_csv('../assays/CTCF-AID_AUX-CTL/analysis_df.csv')

georges_df['Image Name'] = georges_df['Image Name'].apply(lambda x: x[:-9])


# A df with the domains
georges_domains_df = georges_df
georges_domains_df = georges_domains_df.drop(georges_domains_df[georges_domains_df['Channel ID'] == 0].index)
georges_domains_df = georges_domains_df.drop(georges_domains_df[georges_domains_df['roi_type'] == 'subdomain'].index)
georges_domains_df = georges_domains_df.drop(georges_domains_df[georges_domains_df['roi_type'] == 'overlap'].index)
georges_domains_df.dropna(axis=1, how='all', inplace=True)

# A df with the subdomains
georges_subdomains_df = georges_df
georges_subdomains_df = georges_subdomains_df.drop(georges_subdomains_df[georges_subdomains_df['Channel ID'] == 0].index)
georges_subdomains_df = georges_subdomains_df.drop(georges_subdomains_df[georges_subdomains_df['roi_type'] == 'domain'].index)
georges_subdomains_df = georges_subdomains_df.drop(georges_subdomains_df[georges_subdomains_df['roi_type'] == 'overlap'].index)
georges_subdomains_df = georges_subdomains_df['Image Name'].value_counts()


# A df with the overlaps
georges_overlaps_df = georges_df
georges_overlaps_df = georges_overlaps_df.drop(georges_overlaps_df[georges_overlaps_df['Channel ID'] == 0].index)
georges_overlaps_df = georges_overlaps_df.drop(georges_overlaps_df[georges_overlaps_df['roi_type'] == 'subdomain'].index)
georges_overlaps_df = georges_overlaps_df.drop(georges_overlaps_df[georges_overlaps_df['roi_type'] == 'domain'].index)
georges_overlaps_df.dropna(axis=1, how='all', inplace=True)


merged_df = pd.merge(quentin_df, georges_domains_df, on='Image Name', suffixes=('_quentin', '_georges'))
merged_df = pd.merge(merged_df, georges_overlaps_df, on='Image Name', suffixes=('_domain', '_subdomain'))
merged_df.set_index('Image Name', inplace=True)
merged_df["A565_Subdomains_number"] = georges_subdomains_df
merged_df.dropna(axis=1, how='all', inplace=True)


merged_df["Principal_axis_diff"] = merged_df['A565_principal_axis'] - merged_df['major_axis_length']
merged_df["Distance_diff"] = merged_df['distance'] - merged_df['distance3d']
merged_df["Volume_diff"] = merged_df['A565_volume'] - merged_df['volume_domain']
merged_df["Volume_diff_factor"] = merged_df['A565_volume'] / merged_df['volume_domain']
merged_df["A565_CND_number_diff"] = merged_df["A565_CND_number"] - merged_df["A565_Subdomains_number"]

pass