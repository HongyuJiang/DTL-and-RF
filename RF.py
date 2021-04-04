from __future__ import division
import numpy as np
import pandas as pd
from scipy.stats import mode
from DTF import DecisionTreeClassifier

class RandomForestClassifier(object):
    """ A random forest classifier.
    A random forest is a collection of decision trees that vote on a
    classification decision. Each tree is trained with a subset of the data and
    features.
    """

    def __init__(self, n_estimators=32, bootstrap=0.9):
        """
        Args:
            n_estimators: The number of decision trees in the forest.
            max_features: Controls the number of features to randomly consider
                at each split.
            max_depth: The maximum number of levels that the tree can grow
                downwards before forcefully becoming a leaf.
            min_samples_split: The minimum number of samples needed at a node to
                justify a new node split.
            bootstrap: The fraction of randomly choosen data to fit each tree on.
        """
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.forest = []


    def fit(self, data):
        """ Creates a forest of decision trees using a random subset of data and
            features. """
        self.forest = []
        n_samples = len(data)
        n_sub_samples = round(n_samples*self.bootstrap)
        
        for i in range(self.n_estimators):
            data = data.sample(frac=1)
            subset = data[:n_sub_samples]

            tree = DecisionTreeClassifier(subset)
            self.forest.append(tree)


    def predict(self, X):
        """ Predict the class of each sample in X. """
        n_samples = X.shape[0]
        n_trees = len(self.forest)
        predictions = np.empty([n_trees, n_samples])
        for i in range(n_trees):
            predictions[i] = self.forest[i].predict(X)

        return mode(predictions)[0][0]


    def score(self, X, y):
        """ Return the accuracy of the prediction of X compared to y. """
        y_predict = self.predict(X)
        n_samples = len(y)
        correct = 0
        for i in range(n_samples):
            if y_predict[i] == y[i]:
                correct = correct + 1
        accuracy = correct/n_samples
        return accuracy


if __name__ == '__main__':

    dataset = pd.read_csv('wine.csv')
    forest = RandomForestClassifier()
    forest.fit(dataset)

    accuracy = forest.score(dataset.iloc[0, -1], dataset.iloc[-1])
    print('The accuracy was', 100*accuracy, '% on the test data.')

    classifications = forest.predict(dataset.iloc[0, -1])
    print('The digit at index 0 of X_test was classified as a', classifications[0], '.')