from functools import reduce
from operator import add
from pyspark.sql.functions import Column
import pyspark

cancer_classes = None
deaths_column_names = ['Deaths' + str(x) for x in range(1, 27)]


def read_cancer_mortality_data(path):
    data = spark.read\
        .options(header='true', inferSchema='true') \
        .csv(f'{path}/Mort*')
    return data


def read_populations_data(path):
    data = spark.read \
        .options(header='true', inferSchema='true') \
        .csv(path)
    return data


def read_cancer_classes(path):
    global cancer_classes
    with open(path) as f:
        cancer_classes = f.read().splitlines()


def cancer_mortalities():
    data = mortality_data[mortality_data['Cause'].isin(cancer_classes)]
    return data


if __name__ == "__main__":
    spark = pyspark.sql.SparkSession.builder.appName('test').getOrCreate()

    mortality_data = read_cancer_mortality_data(path='../resources/data/codes_replaced')
    populations = read_populations_data(path='../resources/data/codes_replaced/pop_updated.csv')

    # columnstosum = [col(x) for x in deaths_column_names]
    # yearly_mortality = mortality_data.groupBy('Year').sum().collect()

    # Total yearly mortality figures
    # exprs = {x: "sum" for x in deaths_column_names}
    # yearly_mortality = mortality_data.groupBy("Year").agg(exprs)

    ### sum isn't working, but is grouped by year successfully
    yearly_mortality = mortality_data\
        .withColumn('total', sum(mortality_data[col] for col in deaths_column_names))\
        .select('Year', 'total').rdd.collectAsMap()

    # yearly_mortality = mortality_data\
    #     .withColumn('result', sum(mortality_data[col] for col in deaths_column_names))\
    #     .groupBy('Year').collect()

    # yearly_mortality = mortality_data.na.fill(0)\
    #     .withColumn("result", reduce(add, [Column(x) for x in deaths_column_names]))


    read_cancer_classes(path='../resources/data/unique_cancer_class.txt')
    print(cancer_classes)

    cancer_mortalities = mortality_data[mortality_data['Cause'].isin(cancer_classes)]
    print(cancer_mortalities.show())



