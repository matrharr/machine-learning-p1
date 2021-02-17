from sklearn.feature_selection import SequentialFeatureSelector, VarianceThreshold, /
    SelectKBest, chi2, SelectFromModel
from sklearn.ensemble import ExtraTreesClassifier


class DataPreprocessor:

    def __init__(self, dataset, y_label):
        self.data = dataset
        self.x, self.y = self._get_x_y(y_label)


    def _filter_by_variance(self, threshold):
        # removes columns not meeting variance threshold
        sel = VarianceThreshold(threshold=threshold)
        sel.fit_transform(self.x)
        return sel

    def _select_k_best(self, k=5):
        # removes all but k highest scoring features
        x_new = SelectKBest(chi2, k=k).fit_transform(self.x, self.y)
        return x_new

    def _tree_based(self, estimators=50):
        # discard irrelecant features
        clf = ExtraTreesClassifier(n_estimators=estimators)
        clf = clf.fit(self.x, self.y)
        model = SelectFromModel(clf, prefit=True)
        x_new = model.transform(self.x)
        return x_new

    def _sequential_feature_select(self, estimator, n_features, scoring, cv, n_jobs, direction='forward'):
        sfs = SequentialFeatureSelector(
            estimator=estimator,
            n_features_to_select=n_features,
            direction=direction,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs
        )
        sfs.fit(self.x, self.y)
        return sfs

    def _get_x_y(self, y_label):
        pass
