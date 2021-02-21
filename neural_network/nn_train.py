import pandas as pd

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
    
    @staticmethod
    def get_classifer(x, y):
        nn = MLPClassifier(
            solver="adam",
            hidden_layer_sizes=(3,),
            activation='relu',
            batch_size='auto',
            alpha=0.00001, # e-5
            learning_rate='constant',
            learning_rate_init=0.0001,
            power_t=0.5,
            max_iter=300,
            shuffle=True,
            random_state=None,
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
        return [(nn, 'Neural Network', 'neural_network_model')]

    @staticmethod
    def save_figures(clf):
        pass
    # clf = GridSearchCV(MLPClassifier(), PARAMS, cv=3)
    # clf.fit(x, y)
    # print(clf.best_score_)
    # print(clf.best_params_)

    # df = pd.DataFrame(clf.cv_results_)
    # print(df)
