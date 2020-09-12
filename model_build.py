import pandas as pd
import numpy as np
import math
import os


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
    4 Sensitivity
    5. Accuracy

"""
from sklearn import linear_model
from sklearn import svm
from sklearn import naive_bayes
from sklearn.tree import DecisionTreeClassifier
from sklearn import model_selection
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

# Logistic Regression
n_cv = 10
n_Cs = 10
random_state = 101
max_iter = 300
scoring = 'accuracy'
class_weight = 'balanced'
cv_splitter = model_selection.KFold(n_splits=n_cv, shuffle=True, random_state=random_state)
param_grid = {
    'C': [10**uu for uu in np.linspace(-4, 4, n_Cs)]
}

def LRPerformance(X, Y):
        #lr_cv = linear_model.LogisticRegressionCV(cv=n_cv, Cs=n_Cs, random_state=random_state, max_iter=max_iter, scoring=scoring, class_weight=class_weight)
        lr_model = linear_model.LogisticRegression(random_state=random_state, class_weight=class_weight, max_iter=max_iter)
        lr_cv = GridSearchCV(estimator=lr_model, param_grid=param_grid, cv=cv_splitter, scoring=scoring).fit(X,Y)
        return(lr_cv.score(X,Y))

#lr_performance = LRPerformance(X, Y)
#print(lr_performance)

# Support Vector Machine
penalty = 'l1'
loss = 'hinge'
param_grid = {
    'C' : [10**uu for uu in np.linspace(-4, 4, n_Cs)]
}

def SVMPerformance(X, Y):
    svm_model = svm.SVC(random_state=random_state, class_weight=class_weight)
    svm_cv = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=cv_splitter).fit(X, Y)
    return(svm_cv.score(X, Y))

#svm_performance = SVMPerformance(X=train_X, Y=train_Y)
#print(svm_performance)


# Naive Bayes
categorical_cols = ['gender_M', 'specialisation_Mkt&HR', 'workex_Yes']
numerical_cols = ['ssc_p', 'hsc_p', 'degree_p', 'etest_p_trans']
def NBPerformance(X, Y):
    cate_nb = naive_bayes.BernoulliNB().fit(X.loc[:, categorical_cols], Y)
    conti_nb = naive_bayes.GaussianNB().fit(X.loc[:, numerical_cols], Y)
    prob = cate_nb.predict_proba(X.loc[:, categorical_cols]) * conti_nb.predict_proba(X.loc[:, numerical_cols])
    class_predict = list(np.argmax(prob, axis=1))
    class_true  = Y.tolist()
    correct = []
    for i in range(len(class_true)):
        temp = (class_predict[i] == class_true[i])
        correct.append(temp)
    accuracy = sum(correct) / len(class_true)
    return(accuracy)

print(NBPerformance(train_X, train_Y))


# Decision Tree

"""
C. Feature Selection 2nd time

- CV to select best feature set from polynomial of the basic predictors.
"""



"""
D. Hyperparameter Tuning

- Tune to the best hyperparameter
"""
