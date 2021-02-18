import pickle
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_predict, cross_val_score, GridSearchCV, ShuffleSplit
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score

from datasets.load_data import DataLoader
from evaluation.plot_learning_curve import plot_learning_curve
from evaluation.get_metrics import DataMetrics

'''
get training data
'''
# loader = DataLoader('bankrupt')
# bankruptcy = loader.load_data()
# y_col = 'Bankrupt?'
loader = DataLoader('loan')
bankruptcy = loader.load_data()
y_col = 'loan_default'

'''
stratified split - so we split data uniformly
'''
sss = StratifiedShuffleSplit(
    n_splits=2,
    random_state=42,
    test_size=0.2
)
x_all = bankruptcy
y_all = bankruptcy[y_col]

for train_index, test_index in sss.split(x_all, y_all):
    train_set = bankruptcy.loc[train_index]
    test_set = bankruptcy.loc[test_index]

x_train = train_set.drop(y_col, axis=1)
y_train = train_set[y_col].copy()

x_test = test_set.drop(y_col, axis=1)
y_test = test_set[y_col].copy()

'''
check correlation (linear)
'''
corr_matrix = bankruptcy.corr()
print('attribute correlation', corr_matrix[y_col].sort_values(ascending=False))

'''
model
'''
model = DecisionTreeClassifier()
model.fit(x_train, y_train)
print('Training Data Percentage Correct: ', model.score(x_train, y_train))

'''
eval
'''
metrics = DataMetrics().get_metrics(
    model,
    y_train,
    x_train,
    y_all,
    x_all,
    model.predict(x_test),
    y_test,
    mse=True,
    rmse=True,
    confusion_matrix=True,
    cross_val_score=True,
    accuracy_score=True,
    precision_score=True,
    recall_score=True,
    f1_score=True,
    # roc_curve=True,
    roc_auc_score=True
)

'''
learning curve
- plots of the model performance on training and validation sets as a function of the training set size (or iteration). train model several times on different sized subsets of the training set.
'''
fig, axes = plt.subplots(3, 2, figsize=(10, 15))
title = "Learning Curves (Decision Tree)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = model
# plot_learning_curve(
#     estimator,
#     title, 
#     x_train, 
#     y_train, 
#     axes=axes[:, 0], 
#     ylim=(0.7, 1.01),
#     cv=cv, 
#     n_jobs=4
# )

# title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
# # SVC is more expensive so we do a lower number of CV iterations:
# cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
# estimator = SVC(gamma=0.001)
# plot_learning_curve(estimator, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01),
#                     cv=cv, n_jobs=4)
# plt.show()


'''
change threshold for predicting (last attempt to improve performance)
'''


'''
save model
'''
with open('decision_tree_model', 'wb') as f:
    pickle.dump(model, f)


with open('decision_tree_model', 'rb') as f:
    saved_model = pickle.load(f)

print('Training Data Percentage Correct(Saved Model): ', saved_model.score(x_train, y_train))

print('Testing Score', saved_model.score(x_test, y_test))