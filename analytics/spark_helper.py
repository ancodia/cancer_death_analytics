from functools import reduce
from operator import add
from pyspark.sql.functions import Column, udf, array
import pyspark

deaths_column_names = ['Deaths' + str(x) for x in range(1, 27)]


class SparkHelper:

    def __init__(self):
        self.spark = pyspark.sql.SparkSession.builder.appName('cancer_death_analytics').getOrCreate()
        #self.spark.sparkContext.setLogLevel('DEBUG')
        self.cancer_classes = self.read_cancer_classes(path='../resources/data/unique_cancer_class.txt')
        self.mortality_data = self.read_cancer_mortality_data(path='../resources/data/codes_replaced')
        self.cancer_mortalities = self.mortality_data.filter(self.mortality_data['Cause'].isin(self.cancer_classes))
        self.populations = self.read_populations_data(path='../resources/data/codes_replaced/pop_updated.csv')

    def read_cancer_mortality_data(self, path):
        data = self.spark.read\
            .options(header='true', inferSchema='true') \
            .csv(f'{path}/Mort*')
        return data

    def read_populations_data(self, path):
        data = self.spark.read \
            .options(header='true', inferSchema='true') \
            .csv(path)
        return data

    @staticmethod
    def read_cancer_classes(path):
        with open(path) as f:
            classes = f.read().splitlines()
        return classes

    @staticmethod
    def calculate_yearly_total_mortality(df):
        yearly_total = df.withColumn('Total', reduce(add, [df[x] for x in deaths_column_names]))
        return yearly_total

    @staticmethod
    def prepare_yearly_deaths_data(df):
        """
        Returns a DataFrame grouped by year,
        where year is between 1966 and 2016 inclusive.
        """
        # grouped_yearly = df.where((df['Year'] > 1965) & (df['Year'] < 2017)).groupBy('Year')\
        #     .agg({'Total': 'sum'}).orderBy(df['Year'].desc()).toDF('Year', 'Total')
        grouped_yearly = df.groupBy('Year') \
            .agg({'Total': 'sum'}).orderBy(df['Year'].desc()).toDF('Year', 'Total')
        return grouped_yearly

    # @staticmethod
    # def top_n_deaths(n):

# if __name__ == "__main__":
#     spark = pyspark.sql.SparkSession.builder.appName('test').getOrCreate()
#
#     mortality_data = read_cancer_mortality_data(path='../resources/data/codes_replaced')
#     populations = read_populations_data(path='../resources/data/codes_replaced/pop_updated.csv')
#
#
#     mortality_data = calculate_yearly_total_mortality(mortality_data)
#
#     yearly_mortality_data = prepare_yearly_deaths_data(mortality_data)
#     print(yearly_mortality_data)
#
#     read_cancer_classes(path='../resources/data/unique_cancer_class.txt')
#     print(cancer_classes)
#
#     cancer_mortalities = mortality_data[mortality_data['Cause'].isin(cancer_classes)]
#     print(cancer_mortalities.show())
#
#     cancer_mortalities = calculate_yearly_total_mortality(cancer_mortalities)
#
#     yearly_cancer_mortality_data = prepare_yearly_deaths_data(cancer_mortalities)
#     print(yearly_cancer_mortality_data)



