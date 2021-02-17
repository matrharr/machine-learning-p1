import pickle

from sklearn.neighbors import KNeighborsClassifier

from setup_crime_data import CrimeData

# get training data
# does not work with crime data, why?
cd = CrimeData()
x_train, y_train = cd.get_training_data()

# init model
model = KNeighborsClassifier(n_neighbors=5)
# train model
model.fit(x_train, y_train)

'''
fine tune
'''
grid_search = GridSearchCV(model, )
grid_search.fit()
print(grid_search.best_params_)
print(grid_search.best_estimator_)
print(grid_search.cv_results_)

# get % correct
print('Training Data Percentage Correct: ', model.score(x_train, y_train))

# save model
with open('k_nearest_neighbors', 'wb') as f:
    pickle.dump(model, f)


with open('k_nearest_neighbors', 'rb') as f:
    saved_model = pickle.load(f)

print('Training Data Percentage Correct(Saved Model): ', saved_model.score(x_train, y_train))
