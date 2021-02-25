import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV

'''
# of layers
activation function
'''

PARAMS = {
    'epsilon': [1e-08], 
    'learning_rate': ['adaptive'], 
    'learning_rate_init': [0.0001], 
    'solver': ['lbfgs'],
}

class NeuralNetwork:

    def __init__(self):
        self.clfs = []
    
    def get_classifer(self, x, y):
        self.alpha_list = 10.0 ** -np.arange(1, 7)
        # self.alpha_list = [0.001]
        nn_list = []
        for a in self.alpha_list:
            nn = MLPClassifier(
                solver="lbfgs",
                hidden_layer_sizes=(3,2),
                activation='relu',
                batch_size='auto',
                alpha=a, # e-5
                learning_rate='constant',
                learning_rate_init=0.0001,
                power_t=0.5,
                max_iter=400,
                shuffle=True,
                random_state=1,
                tol=0.0001,
                verbose=False,
                warm_start=False,
                momentum=0.9,
                nesterovs_momentum=True,
                early_stopping=False,
                validation_fraction=0.1,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-8,
                n_iter_no_change=10,
                max_fun=15000 # only use w/lbfgs
            )
            nn_list.append((
                nn, f'Neural Network Alpha {a}', f'neural_network_alpha_model'
            ))
        return nn_list

    def save_figures(self, clf):
        self.clfs.append(clf)
        pass

    def plot(self, x_train, y_train, x_test, y_test):
        x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
        train_scores = np.array([clf.score(x_train, y_train) for clf in self.clfs])
        test_scores = np.array([clf.score(x_test, y_test) for clf in self.clfs])

        fig, ax = plt.subplots()
        ax.set_xlabel("alpha")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs alpha for training and testing sets")
        ax.plot(
            self.alpha_list,
            train_scores,
            marker='o',
            label="train",
            drawstyle="steps-post"
        )
        ax.plot(self.alpha_list, test_scores, marker='o', label="test",
                drawstyle="steps-post")
        fig.savefig('figures/nn_acc_vs_alpha.png')
        ax.legend()
        plt.show()
    # clf = GridSearchCV(MLPClassifier(), PARAMS, cv=3)
    # clf.fit(x, y)
    # print(clf.best_score_)
    # print(clf.best_params_)

    # df = pd.DataFrame(clf.cv_results_)
    # print(df)
