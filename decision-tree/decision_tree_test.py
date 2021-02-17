import graphviz
import pickle

from matplotlib import pyplot as plt

from sklearn import tree

from datasets.setup_crime_data import CrimeData

# load trained model
with open('decision_tree_model', 'rb') as f:
    decision_tree_model = pickle.load(f)

# get testing data
cd = CrimeData()
x_test, y_test = cd.get_test_data()

# score on testing data
print('Testing Score', decision_tree_model.score(x_test, y_test))

# plot and save
fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(decision_tree_model, filled=True)
fig.savefig("decision_tree.png")
