# Practical Application III: Comparing Classifiers

**Overview**: In this practical application, your goal is to compare the performance of the classifiers we encountered in this section, namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines.  We will utilize a dataset related to marketing bank products over the telephone.  

**Files**
1. README.md
2. data/bank-additional-full.csv - contains bank marketing campaign data
3. data/bank-additional-names.txt - information about the data set
4. data/bank-additional.csv - contains subset of bank marketing campaign data
4. PortugueseBankMarketingCampaign.ipynb - Jupyter notebook for bank marketing campaign modeling
5. prac3app_utils.py - utilility functions used in the notebook

### Business Understanding

The data is collected from the phone marketing campaign conducted by a Portuguese banking institution. The goal of the campaign is to sell a term term to customers. The business goal is to model the data to successfully predict that a client will subscribe to a term deposit by comparing the following classificatin models:

1. Logistic Regression
2. K Nearest Neighbor
3. Decision Trees
4. Support Vector Machines

### Data Understanding

Input variables:

<b>Bank client data</b><br/>

1 - age (numeric)<br/>
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')<br/>
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)<br/>
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')<br/>
5 - default: has credit in default? (categorical: 'no','yes','unknown')<br/>
6 - housing: has housing loan? (categorical: 'no','yes','unknown')<br/>
7 - loan: has personal loan? (categorical: 'no','yes','unknown')<br/>
<br/>
<b>Related with the last contact of the current campaign</b> <br/>
<br/>
8 - contact: contact communication type (categorical: 'cellular','telephone')<br/>
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec') <br/>
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')<br/>
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model <br/>
<br/>
<b>Other attributes</b><br/>
<br/>
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)<br/>
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)<br/>
14 - previous: number of contacts performed before this campaign and for this client (numeric)<br/>
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')<br/>

<br/>

<b>Social and economic context attributes</b> <br/>
<br/>
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric) <br/>
17 - cons.price.idx: consumer price index - monthly indicator (numeric)<br/>
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)<br/>
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)<br/>
20 - nr.employed: number of employees - quarterly indicator (numeric)<br/>

<br/>
<b>Output variable (desired target)</b><br/>
<br/>
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')<br/>

### Data Analysis and Preparation

1. No null values present in the data.
2. The target variable is skewed towards 'no' 88.73% ofof the custimer did not get a term deposit subscription after the campaign.
3. Numerical Variables
	- 96% of rows for the pdays column has the same value. Therefore, this column can be dropped.
	- 86.34% of rows for the previous column has the same value. Therefore, this column can be dropped.
	- 1.13% of rows are outliers for the column age and about 5.84% of rows are outliers for the column campaign. I am not going to drop those rows as the dataset contains only about 41K rows.
	- duration column can be dropped based on the description of the data.
4. Categorical Variables
	- The count plot and group by plots for all categorical variables look fine
	- The customer is more likely to subscribe to the term deposit when the outcome of the previous campaign is successful (poutcome column )is successful.
	- Certain months lead to success as opposed to others
	- Day of the week, personal loan and housing by themselves don't affect the outcome
5. Apply One Hot Encoding to all categorical Variables and StandardScaler to all numerical variables. 


### Modeling - Compare Classifiers

1. Build four classification models using Logistic Regression, Decision Trees, K nearest Neighbors and Support Vector Classifier.
2. Calculate confusion matrix to understand true positive, false positive, true negative and falsi negative. Calculate precision and recall.
3. Since the primary business objective is to increase successful sales of long-term deposits, we should try to optimize recall.
	- False Positives (predicting a sale when there isn't one): The costs are relatively low, primarily involving some lost effort.
	- False Negatives (predicting no sale when there would have been one): This represents a lost opportunity to acquire a long-term deposit. So we should try to minimize False Negatives.
4. Optimizing recall means focusing on predicting True positives accurately and minimizing False negatives.

#### Evaluation Metrics for different Classifiers

| Model               |   Train Time |   Train Accuracy |   Test Accuracy |   Recall |   Precision |
|:--------------------|-------------:|-----------------:|----------------:|---------:|------------:|
| Logistic Regression |         0.1  |             0.9  |            0.9  |     0.21 |        0.7  |
| Decision Trees      |         0.39 |             1    |            0.84 |     0.34 |        0.31 |
| KNN                 |         0.05 |             0.91 |            0.9  |     0.3  |        0.58 |
| SVC                 |       235.74 |             0.9  |            0.9  |     0.23 |        0.72 |



Decision Tree model has the highest recall but it is also showing high training accuracy which may lead to overfitting. This explains lower test accuracy for Decision Trees as compared to others. recall values are low in general and it can be explained by the imbalanced data set where the 'yes' class is only about 11%


### Hyperparameter Tuning

#### Hyperparameter Tuning

Let's use GridSearchCV to tune hyperparameters.

1. Logistic Regression Hyperparameters
    - C 
    - penalty 
2. Decision Tree Hyperpatameters
   - max depth
    - ceriterion
4. KNearest Neighbors
    - n_neighbors
    - weights
6. SVC (probability is not set to True as it was taking too long to run the grid search)
    - kernel

Only one parameter is chosen for SVC due to compute intensive nature of SVC.


#### Evaluation Metrics for Best Models after GridSearchCV of Classifiers

| Grid Search CV Model   |   Train Time |   Train Accuracy |   Test Accuracy |   Recall |   Precision |
|:-----------------------|-------------:|-----------------:|----------------:|---------:|------------:|
| Logistic Regression    |         6.61 |             0.9  |            0.9  |     0.22 |        0.69 |
| Decision Trees         |         2.14 |             1    |            0.85 |     0.33 |        0.32 |
| KNN                    |        30.19 |             1    |            0.89 |     0.3  |        0.52 |
| SVC                    |        77.3  |             0.91 |            0.9  |     0.24 |        0.68 |


Evaluation metrics are quite similar with and without GridSearchCV except for the following:
1. KNN & SVC precision metrics have reduced.
2. Training accuracy for KNN has gone up by 9 percent.
3. Training times would be higher for GridSearchCV for obvious reasons.


I want to point out that I did not instantiate SVC with probability set to True for grid search to save compute cycles. It was taking too long to run the grid search with that option.

In terms of next steps, I would like to understand the Neural Networks model in detail that is presented in the research paper.
