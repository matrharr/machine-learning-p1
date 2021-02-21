from sklearn.neighbors import KNeighborsClassifier


'''
different values of k
knn clf map
'''

class KNN:

    @staticmethod
    def get_classifer(x, y):
        k_list = [1, 3, 5, 7, 10]
        knn_list = []

        for k in k_list:
            knn_list.append(
                (
                    KNeighborsClassifier(
                        n_neighbors=k
                    ),
                    f'KNN {k}',
                    f'knn_{k}_model'
                )
            )
        return knn_list

    @staticmethod
    def save_figures(clf):
        pass
