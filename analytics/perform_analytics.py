import os
import numpy as np
from matplotlib import ticker
from matplotlib.ticker import FuncFormatter
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
    # Getting dataframe where all causes of death from 1986 and after are included.
    # Total deaths by year is calculated with the 'Deaths1' (deaths for all age groups)
    yearly_all_deaths = mortality_data.filter((mortality_data['Cause'] == 'all')
                                              & (mortality_data['Year'] >= 1986)).groupBy('Year') \
        .agg({'Deaths1': 'sum'}).toDF('Year', 'Total')

    # The same process but using only cancer deaths
    yearly_cancer_deaths = cancer_mortality_data.filter((cancer_mortality_data['Year'] >= 1986)).groupBy('Year') \
        .agg({'Deaths1': 'sum'}).toDF('Year', 'Total')

    # Combining dataframes, subtract cancer deaths from overall to give non-cancer total,
    # calculate the percentage of total deaths which cancer deaths constitute
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
    fig = plt.figure(figsize=(8, 12))  # set plotted figure size
    ax = yearly_deaths.plot.barh(y=['cancer', 'non_cancer'], x='year', stacked=True)
    # display every 5th Year label
    n = 5
    for index, label in enumerate(ax.yaxis.get_ticklabels()):
        if index % n != 0:
            label.set_visible(False)
    plt.ylabel('Year')
    formatter = FuncFormatter(millions)
    ax.xaxis.set_major_formatter(formatter)
    plt.xlabel('Deaths (millions)')
    plt.legend(loc='lower right')
    plt.title('Cancer vs Non-cancer Yearly Deaths')
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    plt.savefig(f'{output_dir}/cancer_vs_non_cancer.png')
    plt.clf()


def linear_regression_to_predict_number_of_deaths():
    """
    Worldwide cancer mortality figures are considered for females aged 20-70 years old.
        > The mean death numbers by age are visualised initially.
        > As the data is curved, a polynomial regression is performed on it with
        the intention of predicting death numbers in the 50-54 age range
    """
    # Relevant columns for 20-70yrs are Deaths[10-19]
    column_index = [str(x) for x in range(10, 20)]

    sum_columns = (', '.join([f'sum(df.Deaths{x})' for x in column_index]))
    cancer_mortality_data.createOrReplaceTempView('df')
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
    fig = plt.figure(figsize=(9, 6))  # set plotted figure size
    plt.xticks(np.arange(len(age_ranges)), age_ranges)
    plt.ylabel('Yearly Average Deaths')
    plt.xlabel('Age')
    plt.title('Cancer Deaths (Female, 20-70 years old)')
    plt.plot(x_axis, y_axis)
    ax = plt.gca()
    ax.yaxis.set_major_formatter(ticker.EngFormatter())
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    plt.savefig(f'{output_dir}/female_cancer_deaths.png')
    plt.clf()

    # split into training and test sets
    (training, test) = female_yearly_totals.randomSplit([.7, .3])
    training.cache()
    test.cache()

    # exclude 'Deaths16' (50-54 years old range) from features in training
    column_index.remove('16')
    vectorised = VectorAssembler(inputCols=[f'Deaths{x}' for x in column_index], outputCol='features')

    poly_expansion = PolynomialExpansion(degree=3, inputCol='features', outputCol='poly_features')

    # set label column to 'Deaths16' as this is the column value being predicted
    lr = LinearRegression(maxIter=10, regParam=0.5).setLabelCol('Deaths16').setPredictionCol('predicted')

    lr_pipeline = Pipeline()
    lr_pipeline.setStages([vectorised, poly_expansion, lr])

    # Fit the model
    model = lr_pipeline.fit(training)

    # predict using test data
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


def most_fatal_cancers_over_time():
    """
    Find and visualise the top 5 most deadly cancers over time.
    """
    # get top 5 causes based on total deaths, ignore other_cancers class
    cancer_mortality_data.createOrReplaceTempView('df')
    top_5_causes = helper.spark.sql('select df.Cause from df '
                                    'where df.Cause!="other_cancers" '
                                    'group by df.Cause '
                                    'order by sum(df.Deaths1) desc limit 5').toDF('cause').collect()
    top_5_causes = [r[0] for r in top_5_causes]
    print(top_5_causes)

    cancer_mortality_data.createOrReplaceTempView('df')
    # get yearly totals
    where = ('or '.join([f'df.Cause="{x}"' for x in top_5_causes]))
    yearly_totals = helper.spark.sql(f'select df.Cause, df.Year, sum(df.Deaths1) from df '
                                     f'where {where} group by df.Cause, df.Year').toDF('cause', 'year', 'total')
    print(yearly_totals.show())

    # plot the total deaths by diagnosis over time
    fig = plt.figure(figsize=(9, 6))  # set plotted figure size

    # create a pandas dataframe for each of the top 5 causes and plot x:year, y:total
    for cause in top_5_causes:
        df = yearly_totals.select('year', 'total').where(yearly_totals['cause'] == cause).orderBy('year').toPandas()
        plt.plot(df['year'], df['total'], label=cause, linewidth=1)
    plt.ylabel('Deaths (millions)')
    plt.xlabel('Year')
    plt.title('Top 5 Most Fatal Cancer Types')
    plt.legend(loc='upper right')
    ax = plt.gca()
    formatter = FuncFormatter(millions)
    ax.yaxis.set_major_formatter(formatter)
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    plt.savefig(f'{output_dir}/top_5.png')
    plt.clf()


def millions(x, pos):
    """
    Used to format tickers on generated graphs
    """
    return '%1.1f' % (x * 1e-6)


if __name__ == "__main__":
    helper = SparkHelper()
    mortality_data = helper.mortality_data
    cancer_mortality_data = helper.cancer_mortality_data

    cancer_vs_non_cancer_deaths_over_time()
    linear_regression_to_predict_number_of_deaths()
    most_fatal_cancers_over_time()
