import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_predict, cross_val_score



class DataMetrics:

    def get_metrics(
        self, model, y_train, x_train, 
        y_all, x_all, predictions, y_test, **kwargs
    ):
        return {
            'mse': self._get_mse(y_test, predictions) if kwargs.get('mse') else None,
            'rmse': self._get_rmse(y_test, predictions) if kwargs.get('rmse') else None,
            'confusion_matrix': self._get_confusion_matrix(model, y_test, x_train, predictions) if kwargs.get('confusion_matrix') else None,
            'cross_val_score': self._get_cross_val_score(model, x_all, y_all) if kwargs.get('cross_val_score') else None,
            'precision_score': self._get_precision_score(y_test, predictions) if kwargs.get('precision_score') else None,
            'recall_score': self._get_recall_score(y_test, predictions) if kwargs.get('recall_score') else None,
            'accuracy_score': self._get_accuracy_score(y_test, predictions) if kwargs.get('accuracy_score') else None,
            'f1_score': self._get_f1_score(y_test, predictions) if kwargs.get('f1_score') else None,
            'roc_curve': self._get_roc_curve(model, y_test, x_train) if kwargs.get('roc_curve') else None,
            'roc_auc_score': self._get_roc_auc_score(y_test, predictions) if kwargs.get('roc_auc_score') else None,
        }

    def _get_mse(self, y_train, predicted):
        mse = mean_squared_error(y_train, predicted)
        print('mse: ', mse)
        return mse

    def _get_rmse(self, y_train, predicted):
        rmse = np.sqrt(self._get_mse(y_train, predicted))
        print('rmse: ', rmse)
        return rmse

    def _get_confusion_matrix(self, model, y_test, x_train, predicted):
        # pred = cross_val_predict(model, x_train, y_train, cv=3)
        conf_mtx = confusion_matrix(y_test, predicted)
        print('confusion matrix: ', conf_mtx)
        return conf_mtx

    def _get_cross_val_score(self, model, x_all, y_all):
        cvs = cross_val_score(model, x_all, y_all, cv=5)
        print('cross validation score: ', cvs)
        print('cross validation mean: ', cvs.mean())
        print('cross validation std: ', cvs.std())
        print('cross validation rmse: ', np.sqrt(-cvs))
        return cvs
    
    def _get_accuracy_score(self, y_test, predictions):
        acc = accuracy_score(y_test, predictions)
        print('accuracy score: ', acc)
        return acc

    def _get_precision_score(self, y_train, predictions):
        ps = precision_score(y_train, predictions)
        print('precision score: ', ps)
        return ps

    def _get_recall_score(self, y_train, predictions):
        rs = recall_score(y_train, predictions)
        print('recall score: ', rs)
        return rs

    def _get_f1_score(self, y_train, predictions):
        f1 = f1_score(y_train, predictions)
        print('f1 score: ', f1)
        return f1

    def _get_roc_curve(self, model, y_train, x_train):
        '''
        find the best threshold
        '''
        pred = cross_val_predict(
            model, x_train, y_train, cv=3, 
            # method="decision_function"
        )
        fpr, tpr, thresholds = roc_curve(y_train, pred)
        self._plot_roc_curve(fpr, tpr, thresholds)
        # print('roc curve: ', rc)
        # return rc

    def _get_roc_auc_score(self, y_test, predictions):
        # pred = cross_val_predict(
        #     model, x_train, y_train, cv=3,
        #     # method="decision_function"
        # )
        ras = roc_auc_score(y_test,predictions)
        print('roc auc score: ', ras)
        return ras

    def _plot_roc_curve(self, fpr, tpr, thresholds):
        plt.plot(fpr, tpr, linewidth=2)
        plt.plot([0,1], [0, 1], 'k--')
        plt.show()