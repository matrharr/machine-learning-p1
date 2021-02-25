import pickle
import sys

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import make_pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_predict, cross_val_score, GridSearchCV, ShuffleSplit
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score

from datasets.load_data import DataLoader
from evaluation.plot_learning_curve import plot_learning_curve
from evaluation.get_metrics import DataMetrics
from boosting.boosting_train import Boosting
from decision_tree.decision_tree_train import DecisionTree
from knn.knn_train import KNN
from neural_network.nn_train import NeuralNetwork
from svm.svm_train import SVM


'''
get training data
# '''
# loader = DataLoader('bankrupt')
# col_trans, dataset = loader.load_data()
# y_col = 'Bankrupt?'
loader = DataLoader('brain')
col_trans, dataset = loader.load_data()
y_col = 'Class'

'''
stratified split - so we split data uniformly
'''
sss = StratifiedShuffleSplit(
    n_splits=2,
    random_state=42,
    test_size=0.3
)

y_all = dataset[y_col]
x_all = dataset

# what is logic in selecting features? compare to correlation
chy = SelectKBest(chi2, k=5)
x_all = chy.fit_transform(x_all, y_all)
print('Selected Columns: ', dataset.columns[chy.get_support()])

for train_index, test_index in sss.split(x_all, y_all):
    train_set = dataset.loc[train_index]
    test_set = dataset.loc[test_index]

x_train = train_set.drop(y_col, axis=1)
y_train = train_set[y_col].copy()

x_test = test_set.drop(y_col, axis=1)
y_test = test_set[y_col].copy()

scaler = preprocessing.StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

'''
check correlation (linear)
'''
corr_matrix = dataset.corr()
print(corr_matrix[y_col].sort_values(ascending=False))

classifiers = [
    # DecisionTree(),
    # KNN(),
    # Boosting(),
    NeuralNetwork(),
    # SVM()
]

for model in classifiers:
    list_clf = model.get_classifer(x_all, y_all)

    for clf, label, name in list_clf:
    
        print('--------------now processing ', label, '-----------------')

        '''
        pipeline
        '''
        pipe = make_pipeline(col_trans, clf)

        '''
        train
        '''
        pipe.fit(x_train, y_train)
        print('Training Data Percentage Correct: ', pipe.score(x_train, y_train))

        '''
        eval
        '''
        metrics = DataMetrics().get_metrics(
            pipe,
            y_train,
            x_train,
            y_all,
            x_all,
            pipe.predict(x_test),
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
        # fig, axes = plt.subplots(3, 2, figsize=(10, 15))
        title = f'Learning Curves ({label})'
        # Cross validation with 100 iterations to get smoother mean test and train
        # score curves, each time with 20% data randomly selected as a validation set.
        cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

        estimator = pipe
        # plot_learning_curve(
        #     estimator,
        #     title,
        #     x_all,
        #     y_all,
        #     axes=axes[:, 0],
        #     ylim=(0.7, 1.01),
        #     cv=cv,
        #     n_jobs=4
        # )

        # fig.savefig(f'figures/{loader.data_name}_{name}_learning_curve')

        # title = r"Learning Curves (SVM, RBF kernel, $\gamma=0.001$)"
        # # SVC is more expensive so we do a lower number of CV iterations:
        # cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
        # estimator = SVC(gamma=0.001)
        # plot_learning_curve(estimator, title, X, y, axes=axes[:, 1], ylim=(0.7, 1.01),
        #                     cv=cv, n_jobs=4)
        # plt.show()

        '''
        change threshold for predicting (last attempt to improve performance)
        - predict_proba()
            show percentage of either option, use for threshold change
        '''

        '''
        classifier specific figures
        '''
        model.save_figures(clf)

        '''
        save model
        '''
        # with open(name, 'wb') as f:
        #     pickle.dump(pipe, f)


        # with open(name, 'rb') as f:
        #     saved_model = pickle.load(f)

        print('Training Data Percentage Correct(Saved Model): ', pipe.score(x_train, y_train))

        print('Testing Score', pipe.score(x_test, y_test))

# dtclass = classifiers[0]
# dtclass.plot_alpha_accuracy(x_train, y_train, x_test, y_test)

# knnclass = classifiers[0]
# knnclass.plot(x_train, y_train, x_test, y_test)

nnclass = classifiers[0]
nnclass.plot(x_train, y_train, x_test, y_test)
