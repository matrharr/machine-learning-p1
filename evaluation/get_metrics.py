import numpy as np

from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_predict, cross_val_score



class DataMetrics:

    def get_metrics(
        self, model, y_train, x_train, 
        y_all, x_all, predictions, **kwargs
    ):
        return {
            'mse': self._get_mse(y_train, predictions) if kwargs['mse'] else None,
            'rmse': self._get_rmse(y_train, predictions) if kwargs['rmse'] else None,
            'confusion_matrix': self._get_confusion_matrix(model, y_train, x_train, predictions) if kwargs['confusion_matrix'] else None,
            'cross_val_score': self._get_cross_val_score(model, x_all, y_all) if kwargs['cross_val_score'] else None,
            'precision_score': self._get_precision_score(y_train, predictions) if kwargs['precision_score'] else None,
            'recall_score': self._get_recall_score(y_train, predictions) if kwargs['recall_score'] else None,
            'roc_curve': self._get_roc_curve(model, y_train, x_train) if kwargs['roc_curve'] else None,
            'roc_auc_score': self._get_roc_auc_score(model, y_train, x_train) if kwargs['roc_auc_score'] else None,
        }

    def _get_mse(self, y_train, predicted):
        mse = mean_squared_error(y_train, predicted)
        print('mse: ', mse)
        return mse

    def _get_rmse(self, y_train, predicted):
        rmse =  np.sqrt(self._get_mse(y_train, predicted))
        print('rmse: ', rmse)
        return rmse

    def _get_confusion_matrix(self, model, y_train, x_train, predicted):
        pred = cross_val_predict(model, x_train, y_train, cv=3)
        conf_mtx = confusion_matrix(y_train, pred)
        print('confusion matrix: ', conf_mtx)
        return conf_mtx

    def _get_cross_val_score(self, model, x_all, y_all):
        cvs = cross_val_score(model, x_all, y_all, cv=3)
        print('cross validation score: ', cvs)
        print('cross validation mean: ', cvs.mean())
        print('cross validation std: ', cvs.std())
        return cvs

    def _get_precision_score(self, y_train, predictions):
        ps = precision_score(y_train, predictions)
        print('precision score: ', ps)
        return ps

    def _get_recall_score(self, y_train, predictions):
        rs = recall_score(y_train, predictions)
        print('recall score: ', rs)
        return rs

    def _get_roc_curve(self, model, y_train, x_train):
        pred = cross_val_predict(
            model, x_train, y_train, cv=3, 
            # method="decision_function"
        )
        rc = roc_curve(y_train, pred)
        print('roc curve: ', rc)
        return rc

    def _get_roc_auc_score(self, model, y_train, x_train):
        pred = cross_val_predict(
            model, x_train, y_train, cv=3, 
            # method="decision_function"
        )
        ras = roc_auc_score(y_train, pred)
        print('roc auc score: ', ras)
        return ras
