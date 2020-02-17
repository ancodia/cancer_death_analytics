from functools import reduce
from operator import sub

import numpy as np
import pandas as pd
from matplotlib import ticker
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.regression import GeneralizedLinearRegression

from analytics.spark_helper import SparkHelper
import pyspark.sql.functions as F
import matplotlib.pyplot as plt

helper = None
mortality_data = None
cancer_mortality_data = None


def cancer_vs_non_cancer_deaths_over_time():
    """
    Calculate the percentage change of cancer deaths compared to overall deaths (1986-2016).
    Represented with a stacked bar chart.
    """
    # 'Deaths1' column contains count of deaths for all age groups
    yearly_all_deaths = mortality_data.where(mortality_data['Cause'] == 'all').groupBy('Year') \
        .agg({'Deaths1': 'sum'}).toDF('Year', 'Total')

    yearly_cancer_deaths = cancer_mortality_data.groupBy('Year') \
        .agg({'Deaths1': 'sum'}).toDF('Year', 'Total')

    # Combine dataframes, subtract cancer deaths from overall,
    # calculate percentage of total that cancer deaths constitute
    yearly_all_deaths.registerTempTable("df")
    yearly_cancer_deaths.registerTempTable("df2")

    yearly_deaths = helper.spark.sql('select df.Year, '
                                     'df.Total-df2.Total, '
                                     'df2.Total, '
                                     'df2.Total/df.Total*100 '
                                     'from df join df2 on df.Year=df2.Year '
                                     'order by df.Year asc').toDF('year', 'non_cancer', 'cancer', 'cancer_percentage')
    print(yearly_deaths.show())

    # calculate cancer deaths percentage change over time
    yearly_deaths = yearly_deaths.toPandas()
    print(yearly_deaths['cancer_percentage'].pct_change())

    # construct horizontal bar chart
    ax = yearly_deaths.plot.barh(y=['cancer', 'non_cancer'], x='year', stacked=True, figsize=(20, 40))
    # display every 5th Year label
    n = 5
    for index, label in enumerate(ax.yaxis.get_ticklabels()):
        if index % n != 0:
            label.set_visible(False)
    plt.ylabel("Years")
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    plt.xlabel("Deaths")
    plt.legend(loc="lower right")
    plt.title("Cancer vs Non-cancer Yearly Deaths")
    plt.show()


def cancer_death_age_gender_correlation():
    """
    Worldwide cancer mortality figures are considered first, followed by a look at Ireland only
    - Male vs Female death counts for 20-80 years old
    """
    male_mortality = cancer_mortality_data.filter((cancer_mortality_data['Sex'] == 1)).cache()
    female_mortality = cancer_mortality_data.filter((cancer_mortality_data['Sex'] == 2)).cache()

    # Relevant columns for 20-80yrs are Deaths[11-22]
    column_index = [str(x) for x in range(11, 22)]

    # get age range strings
    column_names = [helper.age_ranges.select('00').filter(
        (helper.age_ranges['index'] == str(x))).collect()[0][0] for x in column_index]
    # Calculate totals for each age range, then transpose so each age range is a column
    male_mortality_totals = helper\
        .calculate_total_deaths(male_mortality,
                                column_index,
                                column_names)\
        .toPandas().transpose()
    female_mortality_totals = helper \
        .calculate_total_deaths(female_mortality,
                                column_index,
                                column_names) \
        .toPandas().transpose()
    for frame in [male_mortality_totals, female_mortality_totals]:
        frame.columns = ['total']
        frame['range'] = frame.index
    print(male_mortality_totals.head())
    print(female_mortality_totals.head())

    # convert pandas dataframes back to pyspark for analysis
    male_mortality_totals = helper.spark.createDataFrame(male_mortality_totals, ['total', 'range'])
    female_mortality_totals = helper.spark.createDataFrame(female_mortality_totals, ['total', 'range'])
    print(male_mortality_totals.show())
    print(female_mortality_totals.show())

    # features = ["total"]
    # lr_data = male_mortality_totals.select(male_mortality_totals['range'].alias("label"), *features)
    # lr_data.printSchema()
    #
    # lr_data = male_mortality_totals.withColumnRenamed("Total", "label")
    #
    # (training, test) = df.randomSplit([.7, .3])
    # print('train ' + str(training.count()))
    # print('test ' + str(test.count()))
    #
    # vectorAssembler = VectorAssembler(inputCols=['feature'], outputCol='unscaled_features')
    # standardScaler = StandardScaler(inputCol='unscaled_features', outputCol='features')
    # glr = GeneralizedLinearRegression(maxIter=10, regParam=.01).setLabelCol('label').setPredictionCol('Predicted')
    #
    # stages = [vectorAssembler, standardScaler, glr]
    # pipeline = Pipeline(stages=stages)





    vectorizer = VectorAssembler()
    vectorizer.setInputCols(['total'])
    vectorizer.setOutputCol("features")

    glr = GeneralizedLinearRegression(family="gaussian", link="identity", maxIter=10, regParam=0.3)\
        .setPredictionCol("Predicted")\
        .setLabelCol("range")

    glr_pipeline = Pipeline()
    glr_pipeline.setStages([vectorizer, glr])

    # Fit the model
    model = glr_pipeline.fit(male_mortality_totals)

    # Print the coefficients and intercept for generalized linear regression model
    print("Coefficients: " + str(model.coefficients))
    print("Intercept: " + str(model.intercept))

    # Summarize the model over the training set and print out some metrics
    summary = model.summary
    print("Coefficient Standard Errors: " + str(summary.coefficientStandardErrors))
    print("T Values: " + str(summary.tValues))
    print("P Values: " + str(summary.pValues))
    print("Dispersion: " + str(summary.dispersion))
    print("Null Deviance: " + str(summary.nullDeviance))
    print("Residual Degree Of Freedom Null: " + str(summary.residualDegreeOfFreedomNull))
    print("Deviance: " + str(summary.deviance))
    print("Residual Degree Of Freedom: " + str(summary.residualDegreeOfFreedom))
    print("AIC: " + str(summary.aic))
    print("Deviance Residuals: ")
    summary.residuals().show()

    # get age range strings
    # column_names = [helper.age_ranges.select('00').filter(
    #     (helper.age_ranges['index'] == str(x))).collect()[0][0] for x in column_index]
    # exprs = [F.avg(f'mortality_rate{x}') for x in column_index]
    #
    # male_average = male_colorectal_mortality \
    #     .agg(*exprs).toDF(*column_names).toPandas().transpose()
    # female_average = female_colorectal_mortality \
    #     .agg(*exprs).toDF(*column_names).toPandas().transpose()
    #
    # for frame in [male_average, female_average]:
    #     frame.columns = ['value']
    #     frame['range'] = frame.index
    #     print(frame.head())
    #     plt.plot(frame['range'], frame['value'])

    #
    # for frame in [male_average, female_average]:
    #     plt.plot(frame['range'], frame['value'])

    # plt.ylabel("Average Deaths per 100,000")
    # plt.xlabel("Age")
    # plt.legend(loc="upper right")
    # plt.title("Worldwide Cancer Mortality by Age")
    # plt.show()


def most_fatal_cancers_over_time():
    """

    """


if __name__ == "__main__":
    helper = SparkHelper()
    mortality_data = helper.mortality_data
    cancer_mortality_data = helper.cancer_mortality_data

    # cancer_vs_non_cancer_deaths_over_time()
    cancer_death_age_gender_correlation()

    #######################
    dfs_causes = dict()
    for c in helper.cancer_classes:
        dfs_causes[c] = cancer_mortality_data.where(cancer_mortality_data['Cause'] == c)
    # cancer_class_dfs = {k: v for (k, v) in cancer_mortality_data.select('Cause').distinct().collect()}
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
