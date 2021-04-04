import sys
import math
import pandas as pd
import numpy as np
from scipy.stats import entropy
from datetime import datetime

class Node(object):
    def __init__(self, attribute, threshold):
        self.attr = attribute
        self.thres = threshold
        self.left = None
        self.right = None
        self.leaf = False
        self.predict = None

class DecisionTreeClassifier(object):

    def __init__(self, data):
        df_train = data
        attributes =  list(df_train.columns[0: -1])
        self.root = self.build_tree(df_train, attributes, df_train.columns[-1], 3) 

    # First select the threshold of the attribute to split set of test data on
    # The threshold chosen splits the test data such that information gain is maximized
    def select_threshold(self, df, attribute, predict_attr):
        # Convert dataframe column to a list and round each value
        values = df[attribute].tolist()
        values = [ float(x) for x in values]
        # Remove duplicate values by converting the list to a set, then sort the set
        values = set(values)
        values = list(values)
        values.sort()
        max_ig = float("-inf")
        thres_val = 0
        # try all threshold values that are half-way between successive values in this sorted list
        for i in range(0, len(values) - 1):
            thres = (values[i] + values[i+1])/2
            ig = self.info_gain(df, attribute, predict_attr, thres)
            if ig > max_ig:
                max_ig = ig
                thres_val = thres
        # Return the threshold value that maximizes information gained
        return thres_val

    # Calculate info content (entropy) of the test data
    def info_entropy(self, df, predict_attr):
        # Dataframe and number of positive/negatives examples in the data
        low_df = df[df[predict_attr] == 5]
        mid_df = df[df[predict_attr] == 6]
        high_df = df[df[predict_attr] == 7]

        entropy = 0.0 

        if low_df.shape[0] != 0:
            entropy += -1 * float(low_df.shape[0]) * np.log2(float(low_df.shape[0]))
        if mid_df.shape[0] != 0:
            entropy += -1 * float(mid_df.shape[0]) * np.log2(float(mid_df.shape[0]))
        if high_df.shape[0] != 0:
            entropy += -1 * float(high_df.shape[0]) * np.log2(float(high_df.shape[0]))
        # Calculate entropy
        return entropy


    # Calculates the weighted average of the entropy after an attribute test
    def remainder(self, df, df_subsets, predict_attr):
        # number of test data
        num_data = df.shape[0]
        remainder = float(0)
        for df_sub in df_subsets:
            if df_sub.shape[0] > 1:
                remainder += float(df_sub.shape[0]/num_data) * self.info_entropy(df_sub, predict_attr)
        return remainder

    # Calculates the information gain from the attribute test based on a given threshold
    # Note: thresholds can change for the same attribute over time
    def info_gain(self, df, attribute, predict_attr, threshold):
        sub_1 = df[df[attribute] < threshold]
        sub_2 = df[df[attribute] > threshold]
        # Determine information content, and subract remainder of attributes from it
        ig = self.info_entropy(df, predict_attr) - self.remainder(df, [sub_1, sub_2], predict_attr)
        return ig

    # Returns the number of positive and negative data
    def num_class(self, df, predict_attr):

        low_df = df[df[predict_attr] == 5]
        mid_df = df[df[predict_attr] == 6]
        high_df = df[df[predict_attr] == 7]
        return [low_df.shape[0], mid_df.shape[0], high_df.shape[0]]

    # Chooses the attribute and its threshold with the highest info gain
    # from the set of attributes
    def choose_attr(self, df, attributes, predict_attr):
        max_info_gain = float("-inf")
        best_attr = None
        threshold = 0
        # Test each attribute (note attributes maybe be chosen more than once)
        for attr in attributes:
            thres = self.select_threshold(df, attr, predict_attr)
            #print(6, datetime.now())
            ig = self.info_gain(df, attr, predict_attr, thres)
            if ig > max_info_gain:
                max_info_gain = ig
                best_attr = attr
                threshold = thres
        return best_attr, threshold

    # Builds the Decision Tree based on training data, attributes to train on,
    # and a prediction attribute
    def build_tree(self, df, cols, predict_attr, minleaf):
        # Get the number of positive and negative examples in the training data
        nums = self.num_class(df, predict_attr)
        codes = [0, 1, 2]
        # If train data has all positive or all negative values
        # then we have reached the end of our tree
        max_value = max(nums)
        max_index = nums.index(max_value)
        max_counter = 0
        zero_counter = 0

        for n in nums:
            if n == max_value: max_counter += 1 
            if n == 0: zero_counter += 1
        if max_counter == 1: code = codes[max_index]
        else: code = 3

        if zero_counter == 2 or df.shape[0] <= minleaf:
            # Create a leaf node indicating it's prediction
            #print('leaf', nums, max_value, max_index, code)
            print(3, datetime.now())
            leaf = Node(None,None)
            leaf.leaf = True
            leaf.predict = code
            return leaf
        else:
            # Determine attribute and its threshold value with the highest
            # information gain
            best_attr, threshold = self.choose_attr(df, cols, predict_attr)
            # Create internal tree node based on attribute and it's threshold
            tree = Node(best_attr, threshold)
            sub_1 = df[df[best_attr] < threshold]
            sub_2 = df[df[best_attr] > threshold]
            # Recursively build left and right subtree
            tree.left = self.build_tree(sub_1, cols, predict_attr, minleaf)
            tree.right = self.build_tree(sub_2, cols, predict_attr, minleaf)
            return tree

    # Given a instance of a training data, make a prediction of healthy or colic
    # based on the Decision Tree
    # Assumes all data has been cleaned (i.e. no NULL data)
    def predict(self, node, row_df):
        # If we are at a leaf node, return the prediction of the leaf node
        if node.leaf:
            return node.predict
        # Traverse left or right subtree based on instance's data
        if row_df[node.attr] <= node.thres:
            return self.predict(node.left, row_df)
        elif row_df[node.attr] > node.thres:
            return self.predict(node.right, row_df)

# Given a set of data, make a prediction for each instance using the Decision Tree
def test_predictions(model, df):
    trans = [5, 6, 7, 'none']
    num_data = df.shape[0]
    num_correct = 0
    for index, row in df.iterrows():
        prediction = model.predict(model.root, row)
        if trans[prediction] == row['quality']:
            num_correct += 1
    return round(num_correct/num_data, 2)

def main():
    # An example use of 'build_tree' and 'predict'
    dataset = pd.read_csv('wine.csv')
    dataset = dataset.query('quality > 4 and quality < 8')
    DT = DecisionTreeClassifier(dataset)
    print("Accuracy of test data")
    df_test = pd.read_csv('wine.csv')
    print(str(test_predictions(DT, df_test) * 100.0) + '%')

if __name__ == '__main__':
    main()