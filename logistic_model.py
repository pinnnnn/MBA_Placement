import pandas as pd
import numpy as np
import math
import sys
import os

wd = os.path.abspath(os.getcwd())
mydat = pd.read_csv(wd + '/Placement_Data_Full_Class.csv')

# Build Model
### Logistic Regression
from sklearn import linear_model
from sklearn import model_selection
import sklearn.metrics as sk_metrics
import copy

##### Data Preparing
categorical_predictor = ['gender', 'specialisation']
numeric_predictor = ['ssc_p', 'hsc_p', 'degree_p']
response = 'status'
X_dummy = pd.get_dummies(mydat.loc[:, categorical_predictor], drop_first=True)
X = pd.concat([X_dummy, mydat.loc[:, numeric_predictor]], axis=1)
Y = mydat.loc[:, response]
#Y = mydat.loc[:, response].map({'Placed':0, 'Not Placed':1})
train_X, test_X, train_Y, test_Y = model_selection.train_test_split(X, Y, test_size=0.2, random_state=101)

##### Select the Best Hyperparameter C
scorings = ['accuracy','neg_log_loss','f1','roc_auc', sk_metrics.make_scorer(sk_metrics.recall_score, pos_label=-1)]
scoring_name = copy.deepcopy(scorings)
scoring_name[4] = 'specificity'
metrics = []
metrics_builtin = []
for i, scoring in enumerate(scorings):
    print('Now i:', i,'\n')

    LR_model = linear_model.LogisticRegressionCV(cv=5, Cs=12, random_state=101, scoring=scoring, class_weight='Balanced').fit(train_X, train_Y)

    print('CV OK!\n')
    for value in LR_model.scores_.values():
        scores = value
    def myMean(x):
        return np.mean(x)
    mean_score = np.apply_along_axis(func1d=myMean, axis=0, arr=scores)

    best_hypara_id = np.where(LR_model.Cs_ == LR_model.C_)
    #print(mean_score[best_hypara_id])
    #print([round(ma,4) for ma in sorted(mean_score, reverse=True)])
    #for coef_path in LR_model.coefs_paths_.values():
    #    coefs_paths = coef_path

    ##### Fit the model
    def modelMetric(predict_table, event_name):
        # cross: a crosstable with the index being whether the predict is correct or not, and the columns being the labels
        # event_name: the name of  an occuring event
        cross = pd.crosstab(index=predict_table['correct'], columns=predict_table['label'])
        columns = cross.columns.tolist()
        non_event_name = [col for col  in columns if col not in event_name]

        TP = cross.loc['correct', event_name].values[0]
        FP = cross.loc['wrong', non_event_name].values[0]
        TN = cross.loc['correct', non_event_name].values[0]
        FN = cross.loc['wrong', event_name].values[0]

        precision = (TP / (TP+FP)).round(3)
        sensitivity = (TP / (FN+TP)).round(3)
        specificity = (TN / (TN+FP)).round(3)
        accuracy = ((TP+TN) / (TP+TN+FP+FN)).round(3)

        return pd.Series({"precision":precision, "sensitivity":sensitivity, "specificity":specificity, "accuracy":accuracy})

    ####### predict and sort out
    predict_Y = LR_model.predict(test_X)
    predict_table = pd.DataFrame({
        "label": test_Y ,
        "predict": predict_Y})

    predict_table["correct"] = (predict_table['label'] == predict_table['predict']).map({True:'correct', False:'wrong'})
    #print(predict_table['correct'])

    event_name = ['Placed']
    #event_name = [1]
    metric = pd.DataFrame({"value":modelMetric(predict_table, event_name), 'scoring':scoring_name[i]})
    metric = metric.reset_index()
    metrics.append(metric)

    #print('scoring_name:', scoring_name)
    #print(LR_model.coef_)
    #print(LR_model.intercept_)


metrics = pd.concat(metrics, axis=0)
#metrics.to_csv(wd + '/LR_metrics_balanced.csv', index=False)

print(metrics)
print(X.columns)
print(LR_model.coef_)
print(LR_model.intercept_)

# logit(p(Placed)) = -20.684 + 1.022*Male - 0.255*HR + 0.169*ssc_p + 0.069*hsc_p + 0.082*degree_p
# Calculate the log odds ratio and the probability
def LRInterpret(gender_M=0, specialisation_HR=0, ssc_p=50, hsc_p=50, degree_p=50):
    if any([per >100 for per in [ssc_p, hsc_p, degree_p]]):
        sys.exit("Errors: The grades are presented by percentage, which are limited from 0 to 100")
    logit = np.array([gender_M, specialisation_HR, ssc_p, hsc_p, degree_p]).dot(np.array(LR_model.coef_[0])) + LR_model.intercept_
    odds_ratio = math.exp(logit)
    probability = 1/(1+math.exp(-logit))
    return {'logit': logit, 'odds_ratio': odds_ratio, 'probability': probability}

print(LRInterpret(gender_M=1, ssc_p=60, degree_p=99))
print(LRInterpret(gender_M=0, ssc_p=101, degree_p=99))
