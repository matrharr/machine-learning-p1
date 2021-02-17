import pickle

from sklearn.ensemble import AdaBoostClassifier

from setup_crime_data import CrimeData

# get training data
cd = CrimeData()
x_train, y_train = cd.get_training_data()

# init model
model = AdaBoostClassifier()
# train model
model.fit(x_train, y_train)

# get % correct
print('Training Data Percentage Correct: ', model.score(x_train, y_train))

# make prediction
# model.predict([[2,0,1]])
# >> array([some ans])


# save model
with open('adaboost_model', 'wb') as f:
    pickle.dump(model, f)


with open('adaboost_model', 'rb') as f:
    saved_model = pickle.load(f)

print('Training Data Percentage Correct(Saved Model): ', saved_model.score(x_train, y_train))