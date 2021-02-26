from sklearn import svm as svmm

'''
use 2 diff kernel functions
'''

class SVM:

    def __init__(self):
        self.clfs = []

    @staticmethod
    def get_classifer(x, y):
        # svm
        svm = svmm.SVC(
            C=1.0,
            kernel='rbf',
            degree=3,
            gamma='scale',
            coef0=0.0,
            shrinking=True,
            probability=False,
            tol=1e-3,
            cache_size=200,
            class_weight=None,
            verbose=False,
            max_iter=100,
            decision_function_shape='ovr',
            break_ties=False,
            random_state=23
        )
        # linear kernel
        # should scale better than kernel='linear'
        lin = svmm.LinearSVC(
            penalty='l2',
            loss='squared_hinge',
            dual=True,
            tol=1e-4,
            C=1.0,
            multi_class='ovr',
            fit_intercept=True,
            intercept_scaling=1,
            class_weight=None,
            verbose=0,
            random_state=1,
            max_iter=100
        )
        return [
            (svm, 'Support Vector Machine', 'svm_model'),
            (lin, 'Linear SVM', 'lin_svm_model'),
        ]

    def save_figures(self, clf):
        self.clfs.append(clf)

    def plot(self, x_train, y_train, x_test, y_test):
        pass
