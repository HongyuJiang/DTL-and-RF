from __future__ import division
import numpy as np
import pandas as pd
from scipy.stats import mode
from DTF import DecisionTreeClassifier

class RandomForestClassifier(object):

    def __init__(self, n_estimators =  4, bootstrap = 1):
        """
        Args:
            n_estimators: The number of decision trees in the forest.
            bootstrap: The fraction of randomly choosen data to fit each tree on.
        """
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.forest = []


    def fit(self, data):
        """ Creates a forest of decision trees using a random subset of data and
            features. """
        self.forest = []
        #n_samples = len(data)
        #n_sub_samples = round(n_samples*self.bootstrap)
        
        for i in range(self.n_estimators):
            data = data.sample(frac=self.bootstrap)
            #subset = data[:n_sub_samples]
            tree = DecisionTreeClassifier(data, 1)
            self.forest.append(tree)


    def predict(self, X):
        """ Predict the class of each sample in X. """
        n_samples = X.shape[0]
        n_trees = len(self.forest)
        predictions = np.empty([n_trees, n_samples])
        for i in range(n_trees):
            for index, row in X.iterrows():
                predictions[i][index] = self.forest[i].predict(self.forest[i].root, row)

        return mode(predictions)[0][0]


    def score(self, X, y):
        """ Return the accuracy of the prediction of X compared to y. """
        trans = [5, 6, 7, 'unknown']
        y_predict = self.predict(X)
        n_samples = len(y)
        correct = 0
        for i in range(n_samples):
            if trans[int(y_predict[i])] == y[i]:
                correct += 1
        return correct / n_samples


if __name__ == '__main__':

    trainData = pd.read_fwf('train')
    testData = pd.read_fwf('test-sample')
    #print(testData)
    forest = RandomForestClassifier()
    forest.fit(trainData)

    print(trainData.iloc[:, 0: -1])
    print(trainData.iloc[:, -1])

    accuracy = forest.score(trainData.iloc[:, 0: -1], trainData.iloc[:, -1].to_list())
    print('The accuracy was', 100*accuracy, '% on the test data.')

    classifications = forest.predict(testData)
    trans = [5, 6, 7, 'unknown']
    for i in range(len(classifications)):
        print(trans[int(classifications[i])])
    #print('The digit at index 0 of X_test was classified as a', classifications[0], '.')