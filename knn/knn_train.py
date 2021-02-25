import numpy as np
import matplotlib.pyplot as plt


from sklearn.neighbors import KNeighborsClassifier


'''
different values of k
knn clf map
'''

class KNN:

    def __init__(self):
        self.k_list = [1, 5, 10, 15, 20, 30]
        self.clfs = []

    def get_classifer(self, x, y):
        knn_list = []

        for k in self.k_list:
            knn_list.append(
                (
                    KNeighborsClassifier(
                        n_neighbors=k,
                        weights='distance', # vs uniform
                        algorithm='kd_tree',
                        leaf_size=30,
                        p=2,
                        metric='minkowski',
                        metric_params=None,
                        n_jobs=None
                    ),
                    f'KNN {k}',
                    f'knn_{k}_model'
                )
            )
        return knn_list

    def save_figures(self, clf):
        self.clfs.append(clf)

    def plot(self, x_train, y_train, x_test, y_test):
        x_train, y_train, x_test, y_test = np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)
        train_scores = np.array([clf.score(x_train, y_train) for clf in self.clfs])
        test_scores = np.array([clf.score(x_test, y_test) for clf in self.clfs])

        fig, ax = plt.subplots()
        ax.set_xlabel("value of K")
        ax.set_ylabel("accuracy")
        ax.set_title("Accuracy vs K for training and testing sets")
        ax.plot(
            self.k_list,
            train_scores,
            marker='o',
            label="train",
            drawstyle="steps-post"
        )
        ax.plot(self.k_list, test_scores, marker='o', label="test",
                drawstyle="steps-post")
        fig.savefig('figures/knn_accuracy.png')
        ax.legend()
        # plt.show()

