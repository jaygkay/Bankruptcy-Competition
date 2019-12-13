import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression 

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold 
from sklearn.model_selection import cross_val_score 
from sklearn.model_selection import cross_val_predict 
from sklearn.metrics import auc, roc_curve, roc_auc_score
from sklearn.metrics import confusion_matrix

import warnings
warnings.filterwarnings('ignore')

# Load the data
train = pd.read_csv('bankruptcytrainpost.csv')
y_train = pd.DataFrame(train['FAIL'])
x_train = train.drop(y_train, axis = 1)

FS = ['AGE','CLTA','GEAR','LAG','CACL','ROCE','SALES','CHAUD','BIG6']
x_train = pd.DataFrame(x_train[FS])

# Logistic Regression and its Evaluation
import statsmodels.api as sm
logit_model=sm.OLS(y_train,x_train)
result=logit_model.fit()
#result.params
print(result.summary())

# Feature Selection
FS = ['AGE','CLTA','LAG','BIG6']
x_train = pd.DataFrame(x_train[FS])
logit_model=sm.OLS(y_train,x_train)
result=logit_model.fit()
print(result.summary())

# y_test
x_test = pd.read_csv('bankruptcytestpostnolabel.csv')
#x_test[cont] = preprocessing.scale(x_test[cont])
x_test = pd.DataFrame(x_test[FS])

# Logistic Regression Classification
clf = LogisticRegression()
clf.fit(x_train, y_train)
kfold = KFold(n_splits = 5)
cv_result = cross_val_score(clf, x_train, y_train, scoring = 'accuracy', cv = kfold)
print(cv_result.mean())

# Grid-search
param_grid = {"penalty" : ['l1', 'l2'], "C" : np.logspace(-5, 5, 10)}
lrc = GridSearchCV(estimator = clf, param_grid = param_grid, cv = kfold)
lrc.fit(x_train,y_train)
print("Train Accuracy for the tuned model", lrc.best_score_)
bestModel = lrc.best_estimator_
bestModel

# Prediction 
y_index = pd.DataFrame(np.arange(1,17))
y_value = pd.Series(bestModel.predict(x_test), name = 'FAIL')
y_test = pd.concat([y_index, y_value], axis = 1)
y_test
