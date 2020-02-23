# Cancer Death Analytics

## Introduction
Cancer is a disease which  will affect the majority of people in some way during  their lifetime. The goal of this project is to use analytic techniques on data from the WHO Mortality Database (WHOMD)<sup>1</sup> to  aid  in  performing the following tasks relating to cancer mortality:
1)  Determine  the  proportion  of  total  deaths  that  cancerconstitutes over time
2)  Predict  the  number  of  deaths  caused  by  cancer  usingregression
3)  Discover the forms of cancer responsible for the greatest number of deaths

The   analytics   performed   on   the   dataset   in   question   is achieved  with  Spark,  PySpark  in  particular.  For  the  purpose of this project, all analytic processes are run in PySpark local development mode. This approach was decided upon because the  available  local  machine  (Ubuntu  18.04  running  on  RazerBlade  Stealth,  16GB  RAM,  Intel  Core  i7)  offers  sufficient processing power for the volume of data being considered atpresent. Due to the nature of Spark, all Python files produced here could easily be moved to a Spark cluster for distributed analytics  with  the  dataset  stored  on  a  Hadoop  cluster,  for example.

A  description  of  the  source  data  is  featured  in  the  next section.   Following   this   is   a   detailed   breakdown   of   the steps  taken  during  data  pre-processing.  Each  of  the  tasksmentioned above are documented with relevant visualisations and  metrics  included.

## Data Description
The mortality data is in the form of several CSV files (ICD-7,8,9,10) where each record consists of a country code, year, gender, cause of death code and counts for that death type by age range. These mortality figures come from data submitted by  countries  to  the  WHO  based  on  International  Statistical Classification of Diseases (ICD) categories and feature records from between 1950 and 2016. Supplementary reference CSV files for country code and yearly population lookups are also included. The population data is also divided into age ranges. Due  to  the  size  of  the  dataset,  it  has  not  been  included  in the repository. The directory structure of the data in the local repository can be seen in Fig. 1.

![source_data](https://user-images.githubusercontent.com/15143222/75116272-79498f80-565e-11ea-842d-8444122148df.png)

<b>Fig 1.</b> Source data files in local repository

Cause  of  death  code  lookup  files  were  created  for  each ICD  file  based  on  the  codes  found  in  the  documentation provided with the WHOMD. The codes change between ICD versions so a separate lookup is necessary for each source file. The  latest  collection  (ICD-10)  has  introduced  much  greater
detail  in  classification  so  not  all  codes  for  those  are  found in  the  WHOMD  documentation.  A  breakdown  of  additional cancer  classes  were  found<sup>2</sup> and  added  to  the  ICD-10  lookup file.  Another  CSV  lookup  file  was  produced  from  the  agerange  table  found  in  the  WHOMD  documentation  for  use with  the  mortality  and  population  datasets.  These  lookup files are found in resources/icd_code_lookup in the associated repository.

## Data Pre-processing
Pre-processing of the data was achieved with the pandas Python library. The resulting code is found in data_prep/clean_data.py. This section outlines all steps involved in this procedure.

![before_processing](https://user-images.githubusercontent.com/15143222/75116418-2a9cf500-5660-11ea-9a85-36fc8febb392.png)

<b>Fig 2.</b> Sample of mortality data before processing

### Country Codes
A dictionary was created in the get_country_codes_dictionary method to facilitate the replacement of country codes with their corresponding name. This was achieved by simply reading the country code CSV and converting the resulting data frame to a dictionary.

### Mortality Data
The clean_icd_data method provides the necessary steps for pre-processing the ICD mortality data. A source file and related cause of death lookup file are provided as arguments. A dictionary was created from the lookup file CSV and a data frame from the source file. Then map() from pandas was used on the data frame to replace country codes (using the dictionary mentioned in the previous subsection) and causes of death relating to cancer. Codes for "all deaths" were also replaced to enable easy lookup of overall death numbers. In the case of ICD-10 data, there are many obscure cancer diagnoses so a regular expression of ^[C].* was passed to replace any of those which fall outside the main expected classes (Fig. 3) with the label other_cancers. All cancer codes begin with "C".

![cancer_classes](https://user-images.githubusercontent.com/15143222/75116438-50c29500-5660-11ea-9420-45905c126362.png)

<b>Fig 3.</b> Cancer classes extracted from lookup files

Before creating a new CSV file with codes replaced, all remaining null values from the source data were replaced with 0. The files were written to resources/data/codes_replaced, ready for analysis. Fig. 2 and Fig. 4 demonstrate the data content before and after pre-processing respectively.

![after_processing](https://user-images.githubusercontent.com/15143222/75116473-aac35a80-5660-11ea-91ae-a25b43b81af6.png)

<b>Fig 4.</b> Sample of mortality data after processing

### Population Data
A process similar to that used on the mortality data was followed when cleaning the populations data. In the clean_populations_data method pandas was used to map country names to their corresponding code and null values were replaced before writing the updated data frame to a new CSV file. As with the updated mortality files, the cleaned population data was written to resources/data/codes_replaced.

After processing the populations data it was discovered that not all years for all countries have a related population record. Due to this fact, populations are not incorporated in the analytics portion of the project because determining which dates are available was found to be too time-consuming for the project's lifecycle. An incomplete attempt to counteract this issue is the calculate_death_rate_per_100k method of analytics/spark_helper.py.

### Cancer Classes
The final step required before the data was deemed ready for analytics was to extract the unique cancer diagnoses from the multiple ICD lookup files. This was accomplished with create_unique_cancer_diagnosis_lookup_file in which the pandas unique() function was used on a data frame constructed from a concatenation of each of the lookup files to get all distinct values (see Fig. 3). This list of diagnoses was stored as resources/data/unique_cancer_class.txt and is used later for filtering the data being analysed.

## Analytics
This section features implementation details, results and visualisations for each of the analytical applications mentioned in the introduction. Two Python files are used to complete this undertaking, namely analytics/spark_helper.py and analytics/perform_analytics.py. 

The helper class is responsible for creating a Spark session and initialising pyspark data frames containing the cleaned data which was described in the previous section. Mortality data was filtered to include only that from before 2016 because some countries figures are not entirely updated. The file for performing analytics is a runnable script consisting of a method for each of the following analytic tasks called sequentially. Analysis results from the data are predominantly obtained using pyspark with the assistance of \verb|pandas| while visualisations were created with the combination of pandas and matplotlib.

### Proportion of Cancer Deaths Compared to Overall
The intention here is to discover what percentage of total deaths are cancer-specific and how this percentage varies over time. Only data from after 1986 is included so that visualising is cleaner with 30 years to display rather than the possible total of around 60 years.

The first step was to create two pyspark data frames, one containing the death counts for all causes and the second with only cancer deaths included. Spark SQL was used to combine these data frames while also updating the totals count for all causes to exclude cancer deaths and calculating the percentage of deaths that were cancer-related for each year. The resulting data frame is made up of a row per year with columns for year, cancer deaths, non-cancer deaths and cancer deaths percentage.

The data frame was converted to a pandas data frame so that the percentage change function could be utilised and subsequently plotted as a horizontal stacked bar chart  to provide a clear representation of the difference in death numbers. The cancer death percentage change can be seen in Table I and the chart in Fig. 5. From these results, it is evident that the ratio of cancer deaths to non-cancer deaths has remained relatively consistent over the analysed time period with 0.1% between 1999 and 20000 being the greatest fluctuation.

![cancer_vs_non_cancer](https://user-images.githubusercontent.com/15143222/75116552-8916a300-5661-11ea-9329-af520f514427.png)

<b>Fig 5.</b> Number of deaths - cancer vs non-cancer


![Selection_076](https://user-images.githubusercontent.com/15143222/75116627-430e0f00-5662-11ea-95ed-99a192dbeff3.png)

<b>Table I.</b> Yearly cancer deaths percentage change - 1986-2016

### Predictive Model for Cancer Death Numbers
A predictive model for estimating number of deaths is the anticipated outcome for this element of the analysis. The focus of the prediction is female deaths from any form of cancer in the 50-54 years old age range. This group was selected arbitrarily to test the model and could be updated to predict any demographic found in the dataset.

![pearson](https://user-images.githubusercontent.com/15143222/75116711-fe36a800-5662-11ea-9b35-ab472d3cb02e.png)

<b>Fig 6.</b> Pearson Correlation Coefficient results

The first step taken in determining how to model the data was to check the Pearson correlation coefficient between the dependent variable (deaths in 50-54 range i.e Deaths16 column) with each of the independent variables (deaths in each of the other age ranges). Spark SQL was used to create a data frame which contains a yearly death count for women in each available age range. 

![female_cancer_deaths](https://user-images.githubusercontent.com/15143222/75116727-20c8c100-5663-11ea-991e-613bf16243bd.png)

<b>Fig 7.</b> Female cancer deaths by age range - 20-70 years

The Spark DataFrameStatFunctions class was used to calculate the correlation coefficient from the new data frame with results showing that all independent variables have a greater than 95\% correlation with the dependent as can be seen in Fig. 6. This indicates a linear nature to the data. Following this, a graph was generated from the data (Fig. 7) and due to the curved shaped, polynomial rather than simple linear regression was chosen as the model to use for prediction.

To build the polynomial regression model, the dataset was split into training and test sets (70/30 split) and a Spark machine learning pipeline was constructed. Fig. 8 displays the architecture of this pipeline. The VectorAssembler element is used to merge the input columns into a vector. The vectorised features are passed to a PolynomialExpansion instance to expand them into a polynomial space, in this case a third-degree polynomial. When the polynomial features are prepared they  are passed to the LinearRegression object thus completing the pipeline flow. The label column for the regression is set to "Deaths16" because this is the element at the focus of this prediction. At this stage, the pipeline is ready to have data fitted to it for training. 

![pipeline](https://user-images.githubusercontent.com/15143222/75116738-49e95180-5663-11ea-8779-073061fd07c3.png)

<b>Fig 8.</b> Regression pipeline


![Selection_077](https://user-images.githubusercontent.com/15143222/75116782-b2383300-5663-11ea-981f-d290668f9f8e.png)

<b>Table II.</b> Polynomial Regression model metrics



Table II features some metrics related to the trained model. The r^2 value indicates the model is a good fit for the data but the Mean Squared Error (MSE) and Mean Absolute Error (MAE) values are very high. A sample of values predicted with the test dataset is shown in Fig. 9, due to the thousands of deaths in question, the MAE may not be as inaccurate as first thought. Metrics associated with the predictions are found in Table III. Again, r^2 shows an adequate fit for the training data while the other metrics are excessively high. This model may not be the best suited to this particular dataset due to the age ranges being in blocks of 5 years and variations in death numbers over time.

![predictions](https://user-images.githubusercontent.com/15143222/75116787-b95f4100-5663-11ea-8c01-72a45f1b69c5.png)

<b>Fig 9.</b> Sample of predicted values

![Selection_078](https://user-images.githubusercontent.com/15143222/75116784-b401f680-5663-11ea-8ac1-8d1ce203cc73.png)

<b>Table III.</b> Prediction metrics

### Most Fatal Forms of Cancer
The final analysis task performed on the cancer mortality dataset was to find the 5 most fatal specific cancer diagnoses over the available timescale. A Spark SQL query was used to uncover the 5 labels while the "other_cancers" classification was excluded from this task despite having the overall highest number of deaths because it lacks specificity. From this query the highest death-causing cancers returned were stomach, lung, colorectal, breast and lymphatic/haemopoietic. 

\begin{figure}[!h]
\centerline{\includegraphics[width=.5\textwidth]{resources/top_5.png}}
\caption{Cancer diagnoses with highest fatalities}
\label{fatality}
\end{figure}

An aggregate of total deaths by cause per year was calculated with another SQL query on the pyspark data frame. Then to enable plotting of this data it was converted to a pandas data frame. Fig. 10 features the resulting graph. No additional metrics were found for this task because it was concluded that the plotted data offers sufficient explanation.

![top_5](https://user-images.githubusercontent.com/15143222/75116879-91bca880-5664-11ea-9899-192a8a46a6fe.png)

<b>Fig 10.</b> Cancer diagnoses with highest fatalities

## Conclusion
This project offered an excellent opportunity for obtaining useful hands-on experience in executing data analytics tasks through Spark. Choosing Python as the tool for all aspects, from pre-processing to analysis and visualisation, was enlightening of its capabilities. Using pandas proved to be an efficient method for cleaning the dataset while pyspark processed the large CSV files with ease. Overall the visualisations generated from the data were found to be more useful than the analytic results for this particular dataset which is an unfortunate result though it has taught several lessons.

When conducting similar analytic tasks in the future, greater care should be taken in the initial exploration of the data. As work progressed on this project, it was discovered that having only death numbers made it difficult to perform anything above simple analysis. To gain deeper knowledge from cancer deaths, additional data relating to aspects such as overall diagnosis levels or GDP would offer more avenues to explore with machine learning techniques. 

Initially, the intention was to have the entire analytics process running within a distributed environment but due to time constraints this was decided against. With this experience of working with Spark it should be possible with little effort to transfer these analytic methods to such an environment. Finally, the necessity for additional study and understanding of statistical concepts was also noted during the course of the project.


<sup>1</sup> https://www.who.int/healthinfo/statistics/mortality_rawdata/en/

<sup>2</sup> https://training.seer.cancer.gov/icd10cm/neoplasm/c-codes.html
