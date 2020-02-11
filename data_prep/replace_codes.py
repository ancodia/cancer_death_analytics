import csv
import os
import pickle

import numpy
import pandas as pd

lookups_path = '../resources/icd_code_lookup/'
data_path = '../resources/data/mortality_rate/'
mortality_data_path = data_path + 'mortality_figures/'

icd7_mortality_data_path = mortality_data_path + 'MortIcd7'
icd7_lookup_path = lookups_path + 'icd_7_cancer_codes.csv'

icd8_mortality_data_path = mortality_data_path + 'Morticd8'
icd8_lookup_path = lookups_path + 'icd_8_cancer_codes.csv'

icd9_mortality_data_path = mortality_data_path + 'Morticd9'
icd9_lookup_path = lookups_path + 'icd_9_cancer_codes.csv'

icd10_1_mortality_data_path = mortality_data_path + 'Morticd10_part1'
icd10_2_mortality_data_path = mortality_data_path + 'Morticd10_part2'
icd10_lookup_path = lookups_path + 'icd_10_cancer_codes_combined.csv'

country_codes_path = data_path + 'country_codes'
populations_path = data_path + 'pop'

country_codes_dict = None


def get_country_codes_dictionary():
    country_codes_df = pd.read_csv(country_codes_path, header=0, sep=',', index_col=False)
    global country_codes_dict
    country_codes_dict = dict(zip(list(country_codes_df['country']), list(country_codes_df['name'])))


def replace_codes_in_icd_data(source_file_path, lookup_file_path, regex_replace=None):
    """
    Replace cause of death (cancer-related only) and country codes with string value for ease of use
    """
    lookup_df = pd.read_csv(lookup_file_path, header=0, sep=',', index_col=False, usecols=['code', 'class'])
    lookup_dict = dict(zip(list(lookup_df['code']), list(lookup_df['class'])))

    main_df = pd.read_csv(source_file_path, header=0, sep=',', index_col=False)

    main_df['Country'] = main_df['Country'].map(country_codes_dict)
    main_df['Cause'] = main_df['Cause'].map(lookup_dict).fillna(main_df['Cause'])

    if regex_replace is not None:
        main_df['Cause'].replace(regex_replace, 'other_cancers', regex=True, inplace=True)

    output_csv = '../resources/data/codes_replaced/' + \
                 os.path.basename(source_file_path) + '_updated.csv'
    main_df.fillna(0).to_csv(output_csv, encoding='utf-8', index=False, float_format='%.f')
    print(f'Updated csv output to: {output_csv}')


def replace_codes_in_populations_data():
    df = pd.read_csv(populations_path, header=0, sep=',', index_col=False)
    df['Country'] = df['Country'].map(country_codes_dict)

    output_csv = '../resources/data/codes_replaced/' + \
                 os.path.basename(populations_path) + '_updated.csv'
    df.fillna(0).to_csv(output_csv, encoding='utf-8', index=False, float_format='%.f')
    print(f'Updated csv output to: {output_csv}')


def create_unique_cancer_diagnosis_lookup_file():
    """
    Create a file based on unique values in each of the ICD diagnosis lookup files
    for lookup purposes during performance of analytics
    """
    lookup_files = [icd7_lookup_path, icd8_lookup_path, icd9_lookup_path, icd10_lookup_path]
    df = pd.concat((pd.read_csv(f, usecols=range(3), lineterminator='\n') for f in lookup_files), ignore_index=True)
    unique_classes = list(df['class'].unique())
    print(unique_classes)
    output_file = '../resources/data/unique_cancer_class.txt'

    with open(output_file, 'w') as f:
        for item in unique_classes:
            f.write("%s\n" % item)
    print(f'Unique cancer class output to: {output_file}')


if __name__ == "__main__":
    get_country_codes_dictionary()

    replace_codes_in_icd_data(icd7_mortality_data_path, icd7_lookup_path)
    replace_codes_in_icd_data(icd8_mortality_data_path, icd8_lookup_path)
    replace_codes_in_icd_data(icd9_mortality_data_path, icd9_lookup_path)
    replace_codes_in_icd_data(icd10_1_mortality_data_path, icd10_lookup_path, regex_replace='^[C].*')
    replace_codes_in_icd_data(icd10_2_mortality_data_path, icd10_lookup_path, regex_replace='^[C].*')

    replace_codes_in_populations_data()
    create_unique_cancer_diagnosis_lookup_file()
