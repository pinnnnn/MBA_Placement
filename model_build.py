import pandas as pd
import numpy as np
import math
import os
from joblib import dump, load


wd = os.path.abspath(os.getcwd())
mydat = pd.read_csv(wd + '/Placement_Data_Transformed.csv')

"""
A. Feature Selection

- Choose features from EDA result as basic predictors.
"""
from sklearn import model_selection
categorical_predictor = ['gender', 'specialisation', 'workex']
numeric_predictor = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p_trans']
response = 'status'

X_dummy = pd.get_dummies(mydat.loc[:, categorical_predictor], drop_first=True)
X = pd.concat([X_dummy, mydat.loc[:, numeric_predictor]], axis=1)
Y = mydat.loc[:, response].map({'Placed':0, 'Not Placed':1})
train_X, test_X, train_Y, test_Y = model_selection.train_test_split(X, Y, test_size=0.2, random_state=101)


"""
B. Model Selection

- Compare models including
    1. Logistic regression
    2. Support Vector Machine
    3. Naive Bayes
    4. Decision Tree

- Hyperparameters are obtained by 10-fold CV being the one
    * with the best 'accuracy' using 'balanced' scoring metrics *

- Select models based on metrics including
    1. ROC AUC
    2. F1 score
    3. PR curve
    4. Sensitivity
    5. Accuracy

"""
from sklearn import linear_model
from sklearn import svm
from sklearn import naive_bayes
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn import metrics

n_cv = 10
n_Cs = 10
random_state = 101
max_iter = 300
scoring = 'accuracy'
class_weight = 'balanced'
cv_splitter = model_selection.KFold(n_splits=n_cv, shuffle=True, random_state=random_state)


def PerformanceCompare(class_true, class_predict, method, score_predict, exist_score=True):
    if exist_score:
        roc_auc = metrics.roc_auc_score(y_true=class_true, y_score=score_predict)
        pr_score = metrics.average_precision_score(y_true=class_true, y_score=score_predict)
    else:
        roc_auc = float('nan')
        pr_score = float('nan')
    f1_score = metrics.f1_score(y_true=class_true, y_pred=class_predict)
    sensitivity = metrics.recall_score(y_true=class_true, y_pred=class_predict)
    accuracy = metrics.accuracy_score(y_true=class_true, y_pred=class_predict)

    output = pd.DataFrame(np.array([roc_auc, pr_score, f1_score, sensitivity, accuracy]),
        index=['ROC_AUC', 'PR_score', 'F1_score', 'sensitivity', 'accuracy'], columns=method)
    return(output)


# Logistic Regression
def LRPerformance(X, Y):
    param_grid = {
        'C': [10**uu for uu in np.linspace(-4, 4, n_Cs)]
    }

    lr_model = linear_model.LogisticRegression(random_state=random_state, class_weight=class_weight, max_iter=max_iter)
    lr_cv = GridSearchCV(estimator=lr_model, param_grid=param_grid, cv=cv_splitter, scoring=scoring).fit(X,Y)
    Y_predict = lr_cv.predict(X)
    score_predict = lr_cv.predict_proba(X)[:,1]
    best_estimator = lr_cv.best_estimator_

    performance = PerformanceCompare(class_true=Y, class_predict=Y_predict, score_predict=score_predict, method=['LR'])
    return({'performance': performance, 'best_estimator': best_estimator})


# Support Vector Machine
def SVMPerformance(X, Y):
    #penalty = 'l1'
    #loss = 'hinge'
    param_grid = {
        'C' : [10**uu for uu in np.linspace(-4, 4, n_Cs)]
    }
    svm_model = svm.SVC(random_state=random_state, class_weight=class_weight)
    svm_cv = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=cv_splitter, scoring=scoring).fit(X, Y)
    Y_predict = svm_cv.predict(X)
    score_predict = svm_cv.decision_function(X)
    best_estimator = svm_cv.best_estimator_

    performance = PerformanceCompare(class_true=Y, class_predict=Y_predict, score_predict=score_predict, method=['SVM'])
    return({'performance': performance, 'best_estimator': best_estimator})


# Naive Bayes
def NBPerformance(X, Y):
    categorical_cols = ['gender_M', 'specialisation_Mkt&HR', 'workex_Yes']
    numerical_cols = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p_trans']

    cate_nb = naive_bayes.BernoulliNB().fit(X.loc[:, categorical_cols], Y)
    conti_nb = naive_bayes.GaussianNB().fit(X.loc[:, numerical_cols], Y)
    prob = cate_nb.predict_proba(X.loc[:, categorical_cols]) * conti_nb.predict_proba(X.loc[:, numerical_cols])
    Y_predict = list(np.argmax(prob, axis=1))
    score_predict = prob[:,1]

    performance = PerformanceCompare(class_true=Y, class_predict=Y_predict, score_predict=score_predict, method=['NB'])
    return({'performance': performance, 'best_estimator': [cate_nb, conti_nb]})


# Decision Tree
def DTPerformance(X, Y):
    param_grid = {
        'max_leaf_nodes': range(3,15),
        'ccp_alpha': np.linspace(0, 0.05, 5)
    }
    dt_model = DecisionTreeClassifier(random_state=random_state, class_weight=class_weight)
    dt_cv = GridSearchCV(estimator=dt_model, param_grid=param_grid, cv=cv_splitter, scoring=scoring).fit(X, Y)
    Y_predict = dt_cv.predict(X)
    score_predict = dt_cv.predict_proba(X)[:,1]
    best_estimator = dt_cv.best_estimator_

    performance = PerformanceCompare(class_true=Y, class_predict=Y_predict, score_predict=score_predict, method=['DT'])
    return({'performance': performance, 'best_estimator': best_estimator})


#lr_pb = LRPerformance(train_X,train_Y)
#svm_pb = SVMPerformance(train_X, train_Y)
#nb_pb = NBPerformance(train_X, train_Y)
#dt_pb = DTPerformance(train_X, train_Y)

#print(lr_pb['best_estimator'])
#performance_table = pd.concat([lr_pb['performance'],
#    svm_pb['performance'],
#    nb_pb['performance'],
#    dt_pb['performance']], axis=1)
#performance_table = round(performance_table, 3)
#print(performance_table)


"""
C. Feature Selection 2nd time

- CV to select best feature set from polynomial of the basic predictors.
- L1 penalty to eliminate redundant variables.
"""
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import RFE
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt

poly = PolynomialFeatures(2)
train_poly = poly.fit_transform(train_X)
poly_name = poly.get_feature_names(train_X.columns)

stay_id= [i for i, x in enumerate(poly_name) if x not in ['gender_M^2','specialisation_Mkt&HR^2','workex_Yes^2']]
train_poly = train_poly[:,stay_id]
poly_name = [poly_name[i] for i in stay_id]

max_iter = 10000
lr_model = linear_model.LogisticRegression(C=0.36, random_state=random_state, class_weight=class_weight, max_iter=max_iter, solver='liblinear')
rfecv = RFECV(estimator = lr_model, step=1, cv=StratifiedKFold(5), scoring='roc_auc').fit(train_poly, train_Y)

plt.figure()
plt.xlabel('Number of features selected')
plt.ylabel('ROC_AUC score')
plt.plot(range(1, len(rfecv.grid_scores_)+1), rfecv.grid_scores_)
#plt.show()

rfe = RFE(estimator = lr_model, step=1, n_features_to_select=21).fit(train_poly, train_Y)
poly_variables_rfe = [poly_name[id] for id, x in enumerate(rfe.support_) if x]

# Lasso Feature Selection
param_grid = {
    'C': [10**x for x in np.linspace(-4,4,n_Cs)]
}


lr_model = linear_model.LogisticRegression(penalty='l1', random_state=random_state, class_weight=class_weight, max_iter=max_iter, solver='liblinear')
#lr_cv = GridSearchCV(lr_model, param_grid=param_grid, scoring='roc_auc').fit(train_poly, train_Y)
#importance = np.abs(lr_cv.best_estimator_.coef_)[0]

threshold = 1e-5
sfm = SelectFromModel(lr_model, threshold=threshold).fit(train_poly, train_Y)


poly_variables_l1 = [poly_name[id] for id, x in enumerate(sfm.get_support()) if x]

feature_selection_output = pd.DataFrame({
    'Feature Name': poly_name,
    'RFE': rfe.support_*1,
    'L1': sfm.get_support()*1
})
feature_selection_output.to_csv(wd+'/feature_selection_output.csv', index=False)

# Set the final version of FEATURE SET
stay_id = [id for id, x in enumerate(poly_name) if x in poly_variables_rfe]
train_X = train_poly[:, stay_id]

test_poly = poly.transform(test_X)
stay_id = [id for id, x in enumerate(poly.get_feature_names(test_X.columns)) if x in poly_variables_rfe]
test_X = test_poly[:, stay_id]


"""
D. Hyperparameter Tuning

- Tune to the best hyperparameter
"""
param_grid = {
    'C': np.linspace(1e-5, 1e5, 20),
    'class_weight': [None, 'balanced']
}

lr_model = linear_model.LogisticRegression(random_state=random_state, max_iter=max_iter, solver='liblinear')
lr_cv = GridSearchCV(lr_model, param_grid=param_grid, scoring='roc_auc').fit(train_X, train_Y)
print(metrics.confusion_matrix(train_Y, lr_cv.predict(train_X)))

dump(lr_cv, wd+'/final_model.joblib')

print(lr_cv.best_score_)


Y_predict = lr_cv.predict(test_X)

confusion_matrix = metrics.confusion_matrix(y_true=test_Y, y_pred=Y_predict)
tn, fp, fn, tp = confusion_matrix.ravel()
print(confusion_matrix)
print([tn, fp, fn, tp])
#accuracy = metrics.accuracy_score(y_true=test_Y, y_pred=Y_predict)
#sensitivity = metri
