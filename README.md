# Practical Application III: Comparing Classifiers

**Overview**: In this practical application, your goal is to compare the performance of the classifiers we encountered in this section, namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines.  We will utilize a dataset related to marketing bank products over the telephone.  

**Files**
1. README.md
2. data/bank-additional-full.csv - contains bank marketing campaign data
3. data/bank-additional-names.txt - information about the data set
4. data/bank-additional.csv - contains subset of bank marketing campaign data
4. PortugueseBankMarketingCampaign.ipynb - Jupyter notebook for bank marketing campaign modeling
5. prac3app_utils.py - utilility functions used in the notebook

###Business Understanding

The data is collected from the phone marketing campaign conducted by a Portuguese banking institution. The goal of the campaign is to sell a term term to customers. The business goal is to model the data to successfully predict that a client will subscribe to a term deposit by comparing the following classificatin models:

1. Logistic Regression
2. K Nearest Neighbor
3. Decision Trees
4. Support Vector Machines

### Data Understanding

Input variables:

Bank client data<br/>

1 - age (numeric)<br/>
2 - job : type of job (categorical: 'admin.','blue-collar','entrepreneur','housemaid','management','retired','self-employed','services','student','technician','unemployed','unknown')<br/>
3 - marital : marital status (categorical: 'divorced','married','single','unknown'; note: 'divorced' means divorced or widowed)<br/>
4 - education (categorical: 'basic.4y','basic.6y','basic.9y','high.school','illiterate','professional.course','university.degree','unknown')<br/>
5 - default: has credit in default? (categorical: 'no','yes','unknown')<br/>
6 - housing: has housing loan? (categorical: 'no','yes','unknown')<br/>
7 - loan: has personal loan? (categorical: 'no','yes','unknown')<br/>
<br/>
Related with the last contact of the current campaign <br/>
8 - contact: contact communication type (categorical: 'cellular','telephone')<br/>
9 - month: last contact month of year (categorical: 'jan', 'feb', 'mar', ..., 'nov', 'dec') <br/>
10 - day_of_week: last contact day of the week (categorical: 'mon','tue','wed','thu','fri')<br/>
11 - duration: last contact duration, in seconds (numeric). Important note: this attribute highly affects the output target (e.g., if duration=0 then y='no'). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model <br/>
<br/>
Other attributes<br/>
12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)<br/>
13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)<br/>
14 - previous: number of contacts performed before this campaign and for this client (numeric)<br/>
15 - poutcome: outcome of the previous marketing campaign (categorical: 'failure','nonexistent','success')<br/>

<br/>

Social and economic context attributes <br/>
16 - emp.var.rate: employment variation rate - quarterly indicator (numeric) <br/>
17 - cons.price.idx: consumer price index - monthly indicator (numeric)<br/>
18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)<br/>
19 - euribor3m: euribor 3 month rate - daily indicator (numeric)<br/>
20 - nr.employed: number of employees - quarterly indicator (numeric)<br/>

<br/>
Output variable (desired target)
21 - y - has the client subscribed a term deposit? (binary: 'yes','no')<br/>


