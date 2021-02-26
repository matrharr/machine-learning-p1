import pandas as pd
import numpy as np

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

    def __init__(self):
        self.clfs = []

    def get_classifer(self, x, y):
        ccp_alphas = self.get_ccp_alphas(x, y)
        # print('num of ccp alphas ', len(ccp_alphas))
        dtrees = []
        ccp_alphas = [0.003]
        for ccp_alpha in ccp_alphas:
            clf = DecisionTreeClassifier(
                criterion='gini',
                splitter='best',
                max_depth=5,
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
            dtrees.append((
                clf,
                f'Decision Tree CCP Alpha {ccp_alpha}',
                f'decision_tree_model'
            ))
        return dtrees

    def save_figures(self, clf):
        self.clfs.append(clf)
        r = tree.export_text(clf)
        print(r)
        fig = plt.figure(figsize=(25,20))
        tree.plot_tree(clf)
        fig.savefig('figures/decision_tree_figure.png')
        plt.show()
    
    def get_ccp_alphas(self, x, y):
        clf_dummy = DecisionTreeClassifier(random_state=0)
        path = clf_dummy.cost_complexity_pruning_path(x, y)
        ccp_alphas, impurities = path.ccp_alphas, path.impurities
        self.ccp_alphas = ccp_alphas
        # self.plot_impurity_alpha(ccp_alphas, impurities)
        return ccp_alphas

    @staticmethod
    def plot_impurity_alpha(ccp_alphas, impurities):
        fig, ax = plt.subplots()
        ax.plot(ccp_alphas[:-1], impurities[:-1], marker='o', drawstyle="steps-post")
        ax.set_xlabel("effective alpha")
        ax.set_ylabel("total impurity of leaves")
        ax.set_title("Total Impurity vs effective alpha for training set")
        fig.savefig('figures/decision_tree_impurity_vs_alpha.png')
        plt.show()

    def plot_alpha_accuracy(self, x_train, y_train, x_test, y_test):
        x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
        train_scores = np.array([clf.score(x_train, y_train) for clf in self.clfs])
        test_scores = np.array([clf.score(x_test, y_test) for clf in self.clfs])

        fig, ax = plt.subplots()
        ax.set_xlabel("alpha")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs alpha for training and testing sets")
        ax.plot(
            self.ccp_alphas,
            train_scores,
            marker='o',
            label="train",
            drawstyle="steps-post"
        )
        ax.plot(self.ccp_alphas, test_scores, marker='o', label="test",
                drawstyle="steps-post")
        fig.savefig('figures/decision_tree_acc_vs_alpha.png')
        ax.legend()
        plt.show()

    # clf = GridSearchCV(DecisionTreeClassifier(), PARAMS, cv=3)
    # clf.fit(x, y)
    # print(clf.best_score_)
    # print(clf.best_params_)

    # df = pd.DataFrame(clf.cv_results_)
    # print(df)

