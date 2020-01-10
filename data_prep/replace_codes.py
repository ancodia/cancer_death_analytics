import pandas as pd

lookups_path = '../resources/icd_code_lookup/'

def replace_cancer_icd_codes(source_file, lookup_file_name):
    lookup_df = pd.read_csv(str(lookups_path + lookup_file_name),
                            header=0,
                            sep=',',
                            index_col=False,
                            usecols=['code', 'class'])
    print(lookup_df)

if __name__== "__main__":
    replace_cancer_icd_codes(None, 'icd_9_cancer_codes.csv')