import csv
import os
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
icd10_lookup_path = lookups_path + 'icd_10_cancer_codes.csv'

country_codes_path = data_path + 'country_codes'
populations_path = data_path + 'pop'

country_codes_dict = None


def get_country_codes_dictionary():
    country_codes_df = pd.read_csv(country_codes_path, header=0, sep=',', index_col=False)
    global country_codes_dict
    country_codes_dict = dict(zip(list(country_codes_df['country']), list(country_codes_df['name'])))


def replace_codes_in_icd_data(source_file_path, lookup_file_path):
    """
    Replace cause of death (cancer-related only) and country codes with string value for ease of use
    """
    lookup_df = pd.read_csv(lookup_file_path, header=0, sep=',', index_col=False, usecols=['code', 'class'])
    lookup_dict = dict(zip(list(lookup_df['code']), list(lookup_df['class'])))

    main_df = pd.read_csv(source_file_path, header=0, sep=',', index_col=False)

    main_df['Country'] = main_df['Country'].map(country_codes_dict)
    main_df['Cause'] = main_df['Cause'].map(lookup_dict).fillna(main_df['Cause'])

    output_csv = '../resources/data/codes_replaced/' + \
                 os.path.basename(source_file_path) + '_updated.csv'
    main_df.to_csv(output_csv, encoding='utf-8', index=False)
    print(f'Updated csv output to: {output_csv}')


def replace_codes_in_populations_data():
    df = pd.read_csv(populations_path, header=0, sep=',', index_col=False)
    df['Country'] = df['Country'].map(country_codes_dict)

    output_csv = '../resources/data/codes_replaced/' + \
                 os.path.basename(populations_path) + '_updated.csv'
    df.to_csv(output_csv, encoding='utf-8', index=False)
    print(f'Updated csv output to: {output_csv}')


if __name__== "__main__":
    get_country_codes_dictionary()

    #https://www.icd10data.com/ICD10CM/Codes/C00-D49
    #https://www.cms.gov/Medicare/Coding/ICD10/Downloads/2016-Code-Descriptions-in-Tabular-Order.zip

    replace_codes_in_icd_data(icd7_mortality_data_path, icd7_lookup_path)
    replace_codes_in_icd_data(icd8_mortality_data_path, icd8_lookup_path)
    replace_codes_in_icd_data(icd9_mortality_data_path, icd9_lookup_path)
    replace_codes_in_icd_data(icd10_1_mortality_data_path, icd10_lookup_path)
    replace_codes_in_icd_data(icd10_2_mortality_data_path, icd10_lookup_path)

    replace_codes_in_populations_data()