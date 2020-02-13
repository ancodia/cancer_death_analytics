from analytics.spark_helper import SparkHelper
import matplotlib.pyplot as plt

if __name__ == "__main__":
    helper = SparkHelper()

    mortality_data = helper.calculate_yearly_total_mortality(helper.mortality_data)

    yearly_mortality_data = helper.prepare_yearly_deaths_data(mortality_data).toPandas()
    print(yearly_mortality_data)
    # yearly_mortality_data.plot(kind='scatter', x='Year', y='Total')

    print(helper.cancer_classes)

    cancer_mortalities = helper.calculate_yearly_total_mortality(helper.cancer_mortalities)
    print(helper.cancer_mortalities.show())

    # yearly_cancer_mortality_data = helper.prepare_yearly_deaths_data(cancer_mortalities).toPandas()
    # print(yearly_cancer_mortality_data)
    # yearly_cancer_mortality_data.plot(kind='scatter', x='Year', y='Total')

    dfs_causes = dict()
    for c in helper.cancer_classes:
        dfs_causes[c] = cancer_mortalities.where(cancer_mortalities['Cause'] == c)
    # cancer_class_dfs = {k: v for (k, v) in cancer_mortalities.select('Cause').distinct().collect()}
    print(dfs_causes)

    for i in range (0, 4):
        df = dfs_causes[list(dfs_causes.keys())[i]] # temporary, need to find top 5 causes
        df = helper.prepare_yearly_deaths_data(df).toPandas()
        plt.plot(df['Year'], df['Total'])

    plt.show()
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

