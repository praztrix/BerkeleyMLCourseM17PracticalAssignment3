
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
import seaborn as sns
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.metrics import precision_score, recall_score
import time


def analyze_outliers(df, feature, outlier_threshold=1.5):
    Q1 = df[feature].quantile(0.25)
    Q3 = df[feature].quantile(0.75)
    IQR = Q3 - Q1
    print("Max",feature,":", df[feature].max(), 
          "  Median ",feature,":", df[feature].median(), 
          "  Mean ",feature,":", df[feature].mean(), 
          "  Min ",feature,":" ,df[feature].min(),
          "  Q1 ", feature, ":" ,Q1,
          "  Q2 ", feature, ":" ,Q3,
          "  IQR ", feature, ":", IQR)
    # identify outliers
    outliers = df[(df[feature] < Q1 - outlier_threshold * IQR) | (df[feature] > Q3 + outlier_threshold * IQR)]
    print(" Outlier Row Count:", outliers.shape[0], " Outlier Row Percent:", 100*(outliers.shape[0]/df.shape[0]),"%")
    print(" Outlier Max:", outliers[feature].max(), 
          " Outlier Median:", outliers[feature].median(), 
          " Outlier Mean:", outliers[feature].mean(), 
          " Outlier Min:",outliers[feature].min())
    return outliers

def subplot_box_hist(df, target, box_title, hist_title, figsize_len = 15, figsize_height = 4):
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(figsize_len, figsize_height))
    axes[0].boxplot(df[target])
    axes[1].hist(df[target], bins=20)
    axes[0].set_title(box_title)
    axes[0].set_ylabel(target)
    axes[1].set_title(hist_title)
    axes[1].set_xlabel(target)
    axes[1].set_ylabel('count')
    plt.show()

def subplot_box_hist_scatter(df, feature, target, box_title, hist_title, scatter_title):
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))
    axes[0].boxplot(df[feature])
    axes[1].hist(df[feature])
    axes[2].scatter(df[feature], df[target])
    axes[0].set_title(box_title)
    axes[0].set_ylabel(feature)
    axes[1].set_title(hist_title)
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel('count')
    axes[2].set_title(scatter_title)
    axes[2].set_xlabel(feature)
    axes[2].set_ylabel(target)
    plt.show()
    
    
#value count plot with count and percentage labels.
def pretty_value_count_plot(df, col, plt_title="", figsize_len = 20, figsize_height = 6):
    plt.figure(figsize=(figsize_len, figsize_height))
    ax = sns.countplot(x=df[col], order = df[col].value_counts(ascending=False).index)
    if(len(plt_title) > 0):
        ax.set_title(plt_title)
    df_y_count = df[col].value_counts(ascending=False)
    df_y_pcnt  = df[col].value_counts(ascending=False, normalize=True).values*100
    df_y_lbls = [f'{p[0]} ({p[1]:.2f}%)' for p in zip(df_y_count, df_y_pcnt)]
    ax.bar_label(container=ax.containers[0], labels=df_y_lbls)
    plt.show()




def do_cat_cont_feature_independence_anova_test(df, feature, target):
    # Ref: https://www.youtube.com/watch?v=u3Hwt_jbbTc
    #Anova test
    grouped_values = []
    #retrieve unique values for the categorical feature.
    cond_groups = df[feature].unique()
    # for each category of the categorical feature, create a list of target variables and append it to grouped values
    # df[df[feature]==group][target] gives you a list of target variables for a specific group/category
    for group in cond_groups:
        grouped_values.append(df[df[feature]==group][target])
    s, pvalue = stats.f_oneway(*grouped_values)
    #NULL Hypothesis is feature is independent
    if(pvalue < 0.05):
        #reject the null hypothesis
        print(feature, " is likely dependent")
    else:
        print(feature, " is likely independent")
        
def do_cat_cat_feature_independence_chisq_test(feature, target, col1, col2):
    contingency_table = pd.crosstab(col1, col2)
    chi2, pvalue, dof, expected_frequencies = chi2_contingency(contingency_table)
    #NULL Hypothesis is feature is independent
    if(pvalue < 0.05):
        #reject the null hypothesis
        print(feature, " is likely dependent")
    else:
        print(feature, " is likely independent")

def plot_barplot_with_groupby(df, feature, target, figure_len = 20, figure_height = 6):
    plt.figure(figsize=(figure_len,figure_height))
    rename_str = f'term deposit subscription % (column {target})'
    df_percentages = df.groupby(feature)[target].value_counts(normalize=True).mul(100).rename(rename_str).reset_index()
    # Plot the data
    sns.barplot(x=feature, y=rename_str, hue=target, data=df_percentages, palette='dark:red')
    plt.show()

def score_model(pipe, X_test, y_test, model_string):
    preds = pipe.predict(X_test)
    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(conf_matrix)
    disp.plot()
    plt.title(f'Confusion Matrix for {model_string}')
    plt.show()
    fp = conf_matrix[0][1]
    fn = conf_matrix[1][0]
    tp = conf_matrix[1][1]
    tn = conf_matrix[0][0]


    # Calculate recall
    recall = recall_score(y_test, preds, pos_label='yes')
    recall = round(recall, 2)
    
    # Calculate precision
    precision = precision_score(y_test, preds, pos_label='yes')
    precision = round(precision, 2)
    print('fp:', fp, 'fn:', fn, 'tp:', tp, 'tn:', tn)
    print('recall:', recall, ' precision:', precision)
    return fp, fn, tp, tn, auc, precision, recall



def score_model_and_plot_roc_curve(pipe, X_test, y_test, model_string):
    
    from sklearn.metrics import auc as skl_auc
    
    fig, ax = plt.subplots(1, 2)
    fig.set_size_inches(12, 12)
    
    preds = pipe.predict(X_test)
    # Confusion Matrix
    
    conf_matrix = confusion_matrix(y_test, preds)
    disp1 = ConfusionMatrixDisplay(conf_matrix)
    disp1.plot(ax=ax[0])
    ax[0].set_title(f'Confusion Matrix for {model_string} model')
    

    fp = conf_matrix[0][1]
    fn = conf_matrix[1][0]
    tp = conf_matrix[1][1]
    tn = conf_matrix[0][0]


    # Calculate recall
    recall = recall_score(y_test, preds, pos_label='yes')
    recall = round(recall, 2)
    
    # Calculate precision
    precision = precision_score(y_test, preds, pos_label='yes')
    precision = round(precision, 2)
        
    y_score = pipe.predict_proba(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score, pos_label='yes')
    roc_auc_value = skl_auc(fpr, tpr)
    
    disp2 = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc_value)
    disp2.plot(ax=ax[1])
    
   
    auc = round(roc_auc_value, 2)

    print('fp:', fp, 'fn:', fn, 'tp:', tp, 'tn:', tn, 'auc:', auc)
    print('recall:', recall, ' precision:', precision)

    # recall or sensitivity = tp/(tp+fn)
    # precision = tp/(tp+fp)
    # specificity = tn/(tn+fp)
    return fp, fn, tp, tn, auc, precision, recall


def fit_pipe(pipe, X_train, y_train, X_test, y_test):
    start_time = time.time()
    pipe.fit(X_train, y_train)
    end_time = time.time()
    training_time = end_time - start_time
    training_time = round(training_time, 2)

    test_acc = pipe.score(X_test, y_test)
    train_acc = pipe.score(X_train, y_train)

    test_acc = round(test_acc, 2)
    train_acc = round(train_acc, 2)

    return training_time, test_acc, train_acc




