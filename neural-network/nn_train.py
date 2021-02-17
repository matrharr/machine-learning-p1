import pickle

from sklearn.neural_network import MLPClassifier

from setup_crime_data import CrimeData

# get training data
cd = CrimeData()
x_train, y_train = cd.get_training_data()

model = MLPClassifier(
    solver="lbfgs",
    alpha=1e-5,
    hidden_layer_sizes=(5, 2)
)

model.fit(x_train, y_train)

print('Training Data Percentage Correct: ', model.score(x_train, y_train))

# save model
with open('neural_network_model', 'wb') as f:
    pickle.dump(model, f)


with open('neural_network_model', 'rb') as f:
    saved_model = pickle.load(f)

print('Training Data Percentage Correct(Saved Model): ', saved_model.score(x_train, y_train))
