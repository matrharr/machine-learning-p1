import graphviz
import pickle

from matplotlib import pyplot as plt

from sklearn import tree

from setup_crime_data import CrimeData

# load trained model
with open('neural_network_model', 'rb') as f:
    neural_network_model = pickle.load(f)

# get testing data
cd = CrimeData()
x_test, y_test = cd.get_test_data()

# score on testing data
print('Testing Score', neural_network_model.score(x_test, y_test))

