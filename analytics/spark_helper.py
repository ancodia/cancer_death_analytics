from functools import reduce
from operator import add
from pyspark.sql.functions import udf, Column
import pyspark
from pyspark.sql.types import FloatType, IntegerType

deaths_column_names = ['Deaths' + str(x) for x in range(1, 27)]


class SparkHelper:

    def __init__(self):
        self.spark = pyspark.sql.SparkSession.builder.appName('cancer_death_analytics').getOrCreate()
        # self.spark.sparkContext.setLogLevel('DEBUG')
        self.cancer_classes = self.read_cancer_classes(path='../resources/data/unique_cancer_class.txt')
        self.mortality_data = self.read_data(path='../resources/data/codes_replaced/Mort*')
        # filter for data between 1950-2015,
        # exclude UK - England, Wales, Scotland and NI are also reported separately
        self.mortality_data = self.mortality_data \
            .filter((self.mortality_data['Year'] <= 2015)
                    & (self.mortality_data['Country'] != 'United Kingdom'))

        self.cancer_mortality_data = self.mortality_data.filter(self.mortality_data['Cause'].isin(self.cancer_classes))
        self.populations = self.read_data(path='../resources/data/codes_replaced/pop_updated.csv')
        self.age_ranges = self.read_data(path='../resources/age_range_lookup.csv')

    def read_data(self, path):
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

    def calculate_death_rate_per_100k(self, df):
        """
        *** Unused for now, populations dataset requires further processing
        because records do not exist for all years ***
        :param df:
        """
        # join input dataframe with self.populations
        column_index = [str(x) for x in range(1, 27)]

        # some years are not included in the populations dataset so use previous/next years values
        # conditions = [df['Country'] == self.populations['Country'],
        #               df['Sex'] == self.populations['Sex'],
        #               df['Year'] == self.populations['Year']
        #               | df['Year']]

        df.registerTempTable("df")
        self.populations.registerTempTable("df2")

        population_columns = ("".join([f',df2.Pop{x}' for x in column_index]))
        joined_df = self.spark.sql(f'select df.* {population_columns} from df join df2 '
                                   'on df.Country=df2.Country '
                                   'and df.Sex=df2.Sex '
                                   'and df.Year=df2.Year '
                                   'or (df.Year-1)=df2.Year '
                                   'or (df.Year+1)=df2.Year')\
            .toDF(*df.columns, *[f'Pop{x}' for x in column_index])

        # joined_df = df.join(self.populations, conditions) \
        #     .select(df['*'], *[f'Pop{x}' for x in column_index])

        calculate_udf = udf(lambda x, y: (x / y) * (10 ^ 5) if y != 0 else 0, FloatType())

        for c in column_index:
            joined_df = joined_df.withColumn(f'mortality_rate{c}',
                                             calculate_udf(joined_df[f'Deaths{c}'],
                                                           joined_df[f'Pop{c}'])
                                             )
        mortality_rate_df = joined_df.select('Country', 'Admin1', 'SubDiv', 'Year',
                                             'List', 'Cause', 'Sex', 'Frmat',
                                             *[f'mortality_rate{x}' for x in column_index])
        mortality_rate_df = mortality_rate_df.fillna(0).cache()
        return mortality_rate_df

    def calculate_total_deaths(self, df, column_range, result_column_names):
        df.registerTempTable("df")
        deaths_columns = (", ".join([f'sum(Deaths{x})' for x in column_range]))
        df = self.spark.sql(f'select {deaths_columns} from df').toDF(*result_column_names)
        return df

    # @staticmethod
    # def top_n_deaths(n):

