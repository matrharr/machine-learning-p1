import pandas as pd

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import GridSearchCV



'''
pruning
gini vs info gain
'''

PARAMS = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [10, 5, 2, None],
    'min_samples_split': [2],
    'min_samples_leaf': [1],
    'min_weight_fraction_leaf': [0.0],
    'max_features': [None],
    'random_state': [None],
    'max_leaf_nodes': [None],
    'min_impurity_decrease': [0.0],
    'class_weight': [None],
    'ccp_alpha': [0.0]
}

class DecisionTree:

    @staticmethod
    def get_classifer(x, y):
        clf = DecisionTreeClassifier(
            criterion='gini',
            splitter='best',
            max_depth=10,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_features=None,
            random_state=None,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            class_weight=None,
            ccp_alpha=0.0
        )
        return [(clf, 'Decision Tree', 'decision_tree_model')]

    @staticmethod
    def save_figures(clf):
        fig = plt.figure(figsize=(25,20))
        tree.plot_tree(clf)
        fig.savefig('figures/decision_tree_figure.png')

    # clf = GridSearchCV(DecisionTreeClassifier(), PARAMS, cv=3)
    # clf.fit(x, y)
    # print(clf.best_score_)
    # print(clf.best_params_)

    # df = pd.DataFrame(clf.cv_results_)
    # print(df)

