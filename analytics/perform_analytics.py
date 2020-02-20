import os
import numpy as np
from matplotlib import ticker
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, PolynomialExpansion
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

from analytics.spark_helper import SparkHelper
import matplotlib.pyplot as plt

helper = None
mortality_data = None
cancer_mortality_data = None

output_dir = '../output'


def cancer_vs_non_cancer_deaths_over_time():
    """
    Calculate the percentage change of cancer deaths compared to overall deaths (1986-2016).
    Represented with a stacked bar chart.
    """
    # 'Deaths1' column contains count of deaths for all age groups
    yearly_all_deaths = mortality_data.where(mortality_data['Cause'] == 'all'
                                             & (mortality_data['Year'] >= 1986)).groupBy('Year') \
        .agg({'Deaths1': 'sum'}).toDF('Year', 'Total')

    yearly_cancer_deaths = cancer_mortality_data.where((cancer_mortality_data['Year'] >= 1986)).groupBy('Year') \
        .agg({'Deaths1': 'sum'}).toDF('Year', 'Total')

    # Combine dataframes, subtract cancer deaths from overall,
    # calculate percentage of total that cancer deaths constitute
    yearly_all_deaths.createOrReplaceTempView('df')
    yearly_cancer_deaths.createOrReplaceTempView('df2')

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
    ax = yearly_deaths.plot.barh(y=['cancer', 'non_cancer'], x='year', stacked=True)
    # display every 5th Year label
    n = 5
    for index, label in enumerate(ax.yaxis.get_ticklabels()):
        if index % n != 0:
            label.set_visible(False)
    plt.ylabel('Years')
    ax.xaxis.set_major_formatter(ticker.EngFormatter())
    plt.xlabel('Deaths')
    plt.legend(loc='lower right')
    plt.title('Cancer vs Non-cancer Yearly Deaths')

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    plt.savefig(f'{output_dir}/cancer_vs_non_cancer.png')


def linear_regression_to_predict_number_of_deaths():
    """
    Worldwide cancer mortality figures are considered for females aged 20-70 years old.
    > The mean death numbers by age are visualised first.
    > Then a polynomial regression is performed on this data to predict death numbers in the 50-54 age range
    """
    cancer_mortality_data.createOrReplaceTempView('df')

    # Relevant columns for 20-70yrs are Deaths[10-19]
    column_index = [str(x) for x in range(10, 20)]

    sum_columns = (', '.join([f'sum(Deaths{x})' for x in column_index]))
    female_yearly_totals = helper.spark.sql(f'select df.Year, {sum_columns} from df '
                                            'where df.Sex=2 '
                                            'group by df.Year '
                                            'order by df.Year asc').toDF('Year', *[f'Deaths{x}' for x in column_index])
    print(female_yearly_totals.show())

    # Check Pearson Correlation Coefficient between independent variables and target variable (Deaths16)
    for i in female_yearly_totals.columns:
        if i != 'Year':
            correlation = female_yearly_totals.stat.corr('Deaths16', i)
            print(f'Correlation between {i} and Deaths16: {correlation}')

    # plot mean deaths by age to find out the shape of this dataset
    plot_df = female_yearly_totals.toPandas()
    plot_df.drop(columns=['Year'], inplace=True)  # drop the Year column as it's not needed for this graph
    x_axis = [x for x in range(0, len(plot_df.columns))]
    y_axis = [int(plot_df[y].mean()) for y in plot_df.columns]

    # get age range labels
    age_ranges = [helper.age_ranges.select('00').filter(
        (helper.age_ranges['index'] == str(x))).collect()[0][0] for x in column_index]

    plt.xticks(np.arange(len(age_ranges)), age_ranges)
    plt.ylabel('Yearly Average Number of Deaths')
    plt.xlabel('Age')
    plt.title('Cancer Deaths (Female, 20-70 years old)')
    plt.plot(x_axis, y_axis)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    plt.savefig(f'{output_dir}/female_cancer_deaths.png')

    # female_mortality = cancer_mortality_data.filter((cancer_mortality_data['Sex'] == 2))
    # print(female_mortality.describe())
    (training, test) = female_yearly_totals.randomSplit([.7, .3])
    training.cache()
    test.cache()

    # exclude 'Deaths16' (50-54 years old range) from features in training
    # column_index.remove('16')
    vectorised = VectorAssembler(inputCols=[f'Deaths{x}' for x in column_index], outputCol='features')

    poly_expansion = PolynomialExpansion(degree=3, inputCol='features', outputCol='poly_features')

    # set label column to 'Deaths16' as this is the column value being predicted
    lr = LinearRegression(maxIter=10, regParam=0.5).setLabelCol('Deaths16').setPredictionCol('predicted')

    lr_pipeline = Pipeline()
    lr_pipeline.setStages([vectorised, poly_expansion, lr])

    # Fit the model
    model = lr_pipeline.fit(training)

    predictions = model.transform(test).select('Year', 'Deaths16', 'poly_features', 'predicted')
    print(predictions.show())

    model_details = model.stages[2]
    print('_____________\nModel details:\n_____________')
    # Print the coefficients and intercept for generalized linear regression model
    print('Coefficients: ' + str(model_details.coefficients))
    print('Intercept: ' + str(model_details.intercept))

    # Summarize the model over the training set and print out some metrics
    summary = model_details.summary
    print('Coefficient Standard Errors: ' + str(summary.coefficientStandardErrors))
    print('T Values: ' + str(summary.tValues))
    print('P Values: ' + str(summary.pValues))
    print('r^2: ' + str(summary.r2))
    print('Mean Squared Error: ' + str(summary.meanSquaredError))
    print('Mean Absolute Error: ' + str(summary.meanAbsoluteError))
    print('Explained variance: ' + str(summary.explainedVariance))
    print('Degrees Of Freedom: ' + str(summary.degreesOfFreedom))
    print('Deviance Residuals: ' + str(summary.devianceResiduals))

    # Evaluation metrics for test dataset
    # Create an RMSE evaluator using the label and predicted columns
    reg_eval = RegressionEvaluator(predictionCol='predicted', labelCol='Deaths16', metricName='rmse')

    # Run the evaluator on the DataFrame
    print('_____________\nPrediction evaluation:\n_____________')
    rmse = reg_eval.evaluate(predictions)
    print(f'Root Mean Squared Error: {rmse}')

    # Mean Square Error
    mse = reg_eval.evaluate(predictions, {reg_eval.metricName: 'mse'})
    print(f'Mean Square Error: {mse}')

    # Mean Absolute Error
    mae = reg_eval.evaluate(predictions, {reg_eval.metricName: 'mae'})
    print(f'Mean Absolute Error: {mae}')

    # r2 - coefficient of determination
    r2 = reg_eval.evaluate(predictions, {reg_eval.metricName: 'r2'})
    print(f'r^2: {r2}')
    print('a')


def most_fatal_cancers_over_time():
    """
    Find and visualise the top 5 most deadly cancers over time.
    """
    cancer_mortality_data.createOrReplaceTempView('df')

    # get top 5 causes based on total deaths, ignore other_cancers class
    top_5_causes = helper.spark.sql('select df.Cause from df '
                                    'where df.Cause!="other_cancers" '
                                    'group by df.Cause '
                                    'order by sum(df.Deaths1) desc limit 5').toDF('cause').collect()
    top_5_causes = [r[0] for r in top_5_causes]
    print(top_5_causes)

    cancer_mortality_data.createOrReplaceTempView('df')
    # get totals
    where = ('or '.join([f'df.Cause="{x}"' for x in top_5_causes]))
    yearly_totals = helper.spark.sql(f'select df.Cause, df.Year, sum(df.Deaths1) from df '
                                     f'where {where} group by df.Cause, df.Year').toDF('cause', 'year', 'total')

    # yearly_totals = cancer_mortality_data.select('Cause', 'Year', 'Deaths1')\
    #     .orderBy('Year').filter(cancer_mortality_data['Cause'].isin(top_5_causes))\
    #     .groupBy('Year').agg({'Deaths1': 'sum'})
        #.toDF('cause', 'year', 'total')

    # yearly_totals = cancer_mortality_data.select('Year', 'Cause', 'sum(Deaths1)') \
    #     .filter(cancer_mortality_data['Cause'].isin(top_5_causes)) \
    #     .groupBy('Year') \
    #     .toDF('year', 'cause', 'total')

    print(yearly_totals.show())

    dfs_causes = dict()
    for c in top_5_causes:
        dfs_causes[c] = yearly_totals.select('year', 'total').where(yearly_totals['cause'] == c)


if __name__ == "__main__":
    helper = SparkHelper()
    mortality_data = helper.mortality_data
    cancer_mortality_data = helper.cancer_mortality_data

    # cancer_vs_non_cancer_deaths_over_time()
    # linear_regression_to_predict_number_of_deaths()
    most_fatal_cancers_over_time()

    #######################
    dfs_causes = dict()
    for c in helper.cancer_classes:
        dfs_causes[c] = cancer_mortality_data.where(cancer_mortality_data['Cause'] == c)
    # cancer_class_dfs = {k: v for (k, v) in cancer_mortality_data.select('Cause').distinct().collect()}
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
