from sklearn import svm as svmm

'''
use 2 diff kernel functions
'''

class SVM:

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
            max_iter=1,
            decision_function_shape='ovr',
            break_ties=False,
            random_state=None
        ) # doesnt work
        # linear kernel
        lin = svmm.LinearSVC()
        return [
            (lin, 'Linear SVM', 'lin_svm_model'),
            (svm, 'Support Vector Machine', 'svm_model')
        ]

    @staticmethod
    def save_figures(clf):
        pass
