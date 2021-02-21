from sklearn.ensemble import AdaBoostClassifier


'''
pruning
'''

class Boosting:

    @staticmethod
    def get_classifer(x, y):
        b = AdaBoostClassifier()
        return [(b, 'Boosting', 'boosting_model')]

    @staticmethod
    def save_figures(clf):
        pass
