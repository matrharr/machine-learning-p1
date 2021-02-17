import pickle

from sklearn import svm

from setup_crime_data import CrimeData

# get training data
# does not do well on crimedata
cd = CrimeData()
x_train, y_train = cd.get_training_data()

# init model
model = svm.SVC()
# train model
model.fit(x_train, y_train)

# get % correct
print('Training Data Percentage Correct: ', model.score(x_train, y_train))

# make prediction
# model.predict([[2,0,1]])
# >> array([some ans])


# save model
with open('svm_model', 'wb') as f:
    pickle.dump(model, f)


with open('svm_model', 'rb') as f:
    saved_model = pickle.load(f)

print('Training Data Percentage Correct(Saved Model): ', saved_model.score(x_train, y_train))