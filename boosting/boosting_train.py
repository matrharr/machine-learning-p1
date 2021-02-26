from sklearn.ensemble import AdaBoostClassifier


'''
pruning
'''

class Boosting:

    def __init__(self):
        self.clfs = []

    def get_classifer(self, x, y):
        b = AdaBoostClassifier(
            base_estimator=None,
            n_estimators=50,
            learning_rate=0.5,
            algorithm='SAMME.R',
            random_state=23
        )
        return [(b, 'Boosting', 'boosting_model')]

    def save_figures(self, clf):
        self.clfs.append(clf)
        pass

    
    def plot(self, x_train, y_train, x_test, y_test):
        pass