from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import LinearRegression

from analytics.spark_helper import SparkHelper
import pyspark.sql.functions as F
import matplotlib.pyplot as plt

if __name__ == "__main__":
    helper = SparkHelper()

    mortality_data = helper.mortality_data

    # yearly_all_deaths = mortality_data.where(mortality_data['Cause'] == 'all').groupBy('Year') \
    #     .agg({'Deaths1': 'sum'}).orderBy(mortality_data['Year'].desc()).toDF('Year', 'Total')

    # yearly_mortality_data = helper.prepare_yearly_deaths_data(mortality_data).toPandas()
    # print(yearly_all_deaths.show(100))
    # yearly_mortality_data.plot(kind='scatter', x='Year', y='Total')

    # print(helper.cancer_classes)

    cancer_mortalities = helper.cancer_mortalities

    mortality_rate = helper.calculate_death_rate_per_100k(cancer_mortalities)

    # yearly_cancer_deaths = cancer_mortalities.groupBy('Year', 'Cause') \
    #     .agg({'Deaths1': 'sum'}).orderBy(cancer_mortalities['Year'].desc()).toDF('Year', 'Cause', 'Total')
    #
    # print(yearly_cancer_deaths.show(1000))

    # colorectal cancer male vs female from 20 to 80 years
    male_colorectal_mortality = mortality_rate.filter((mortality_rate['Sex'] == 1)
                                                      & (mortality_rate['Cause'] == 'colorectal'))
    female_colorectal_mortality = mortality_rate.filter((mortality_rate['Sex'] == 2)
                                                        & (mortality_rate['Cause'] == 'colorectal'))

    print(male_colorectal_mortality.show())
    print(female_colorectal_mortality.show())

    # average deaths per 1000 by age
    column_index = [str(x) for x in range(10, 22)]
    # get age range strings
    column_names = [helper.age_ranges.select('00').filter(
        (helper.age_ranges['index'] == str(x))).collect()[0][0] for x in column_index]
    exprs = [F.avg(f'mortality_rate{x}') for x in column_index]

    male_average = male_colorectal_mortality \
        .agg(*exprs).toDF(*column_names).toPandas().transpose()
    female_average = female_colorectal_mortality \
        .agg(*exprs).toDF(*column_names).toPandas().transpose()

    print(male_average.show())
    print(female_average.head())

    # yearly_cancer_mortality_data = helper.prepare_yearly_deaths_data(cancer_mortalities).toPandas()
    # print(yearly_cancer_mortality_data)
    # yearly_cancer_mortality_data.plot(kind='scatter', x='Year', y='Total')

    #######################
    dfs_causes = dict()
    for c in helper.cancer_classes:
        dfs_causes[c] = cancer_mortalities.where(cancer_mortalities['Cause'] == c)
    # cancer_class_dfs = {k: v for (k, v) in cancer_mortalities.select('Cause').distinct().collect()}
    print(dfs_causes)
    #
    # for i in range (0, 4):
    #     df = dfs_causes[list(dfs_causes.keys())[i]] # temporary, need to find top 5 causes
    #     df = helper.prepare_yearly_deaths_data(df).toPandas()
    #     plt.plot(df['Year'], df['Total'])
    #
    # plt.show()
    ######################

    #### Cancer vs non-cancer - probably won't use
    # yearly_mortality_data['Total'] = (yearly_mortality_data['Total']-yearly_cancer_mortality_data['Total'])/1000000
    # yearly_cancer_mortality_data['Total'] = yearly_cancer_mortality_data['Total']/1000000
    # print(yearly_mortality_data)
    # for frame in [yearly_mortality_data, yearly_cancer_mortality_data]:
    #     plt.plot(frame['Year'], frame['Total'])
    #
    # plt.ylabel("Total Deaths (millions)")
    # plt.xlabel("Countries")
    # plt.legend(loc="upper right")
    # plt.title("Cancer vs Non-cancer related deaths")
    # plt.show()

    # df = dfs_causes[list(dfs_causes.keys())[0]]
    #
    # features = ["Total"]
    # lr_data = df.select(df['Year'].alias("label"), *features)
    # lr_data.printSchema()

    # lr_data = df.withColumnRenamed("Total", "label")

    # (training, test) = df.randomSplit([.7, .3])
    # print('train ' + str(training.count()))
    # print('test ' + str(test.count()))
    #
    # vectorAssembler = VectorAssembler(inputCols=features, outputCol="unscaled_features")
    # standardScaler = StandardScaler(inputCol="unscaled_features", outputCol="features")
    # lr = LinearRegression(maxIter=10, regParam=.01).setLabelCol('Total').setPredictionCol('Predicted')
    #
    # stages = [vectorAssembler, standardScaler, lr]
    # pipeline = Pipeline(stages=stages)
    #
    # model = pipeline.fit(training)
    # prediction = model.transform(test).select('Country', 'Total', 'Predicted')
    #
    # print(prediction.count())
    #
    # print(prediction)
    #
    # print(prediction.show(200))
