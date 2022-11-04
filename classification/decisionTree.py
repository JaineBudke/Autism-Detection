from random import seed
from random import randrange
from csv import reader
import pandas as pd
import math

################################
#                              #
#   Decision Tree Algorithm    #
#                              #
################################
# Adapted from: https://machinelearningmastery.com/implement-decision-tree-algorithm-scratch-python/
class DecisionTree:

    max_depth = 1
    min_size  = 1

    # self.max_depth: maximum depth of the tree 
    # self.min_size: minimum number of rows per node
    def __init__(self, max_depth, min_size):
        self.max_depth = max_depth
        self.min_size  = min_size


    # Calculate the Entropy for a split dataset
    def entropy(self, groups, classes):
        # count all samples on groups
        n_instances = 0.0
        for group in groups:
            n_instances += len(group)

        # sum weighted Entropy for each group
        entropy = 0.0
        for group in groups:
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                # p = probability class
                p = [row[-1] for row in group].count(class_val) / size
                if(p != 0):
                    score += (-1) * ( p * math.log(p,2) )
            # weight the group score by its relative size
            entropy += score * (size / n_instances)
        return entropy



    # Calculate the Gini index for a split dataset
    def gini_index(self, groups, classes):
        # count all samples on groups
        n_instances = 0.0
        for group in groups:
            n_instances += len(group)

        # sum weighted Gini index for each group
        gini = 0.0
        for group in groups:
            size = float(len(group))

            # avoid divide by zero
            if size == 0:
                continue
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
        
                # p = probability class
                p = [row[-1] for row in group].count(class_val) / size

                score += p * p
               
            # weight the group score by its relative size
            gini += (1.0 - score) * (size / n_instances)

        return gini


    # Split a dataset based on an attribute and an attribute value
    def test_split(self, column_index, value, dataset):
        left, right = list(), list()
        # for each row on dataset
        for row in dataset:
            
            # create a binary tree. Right: greater values. Left: lower values 
            if row[column_index] < value:
                left.append(row)
            else:
                right.append(row)
            
        return left, right


    # Select the best split point for a dataset
    def get_split(self, dataset, criterion):
        # get list with classes 
        class_values = list(set(row[-1] for row in dataset))

        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        # number of features in the dataset
        features_count = len(dataset[0])-1
        # loop on features
        # these two loops find the best feature to split 
        # and also the values to decide between the child nodes (>3.4 ou <2.7)
        for column_index in range(features_count):
            
            for row in dataset:

                # create two groups: one with values less than the decision value and one with values greater 
                groups = self.test_split(column_index, row[column_index], dataset)  
                
                # calculate the "impurity" with these groups
                if( criterion == "gini" ):
                    gini = self.gini_index(groups, class_values)
                else:
                    gini = self.entropy(groups, class_values)

                # check for all dataset values, if current is the best, set the variables 
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = column_index, row[column_index], gini, groups

        return {'index':b_index, 'value':b_value, 'groups':b_groups}

    # Create a terminal node value
    def to_terminal(self, group):
        # get class the most frequent class in the group  
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)


    # Create child splits for a node or make terminal
    # node: a dictionary that contain index (splitted point), value (to split) and groups (left and right)
    def split(self, node, depth, criterion):
        left, right = node['groups']

        del(node['groups'])
        # check for a no split (empty list)
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        # check for max depth
        if depth >= self.max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # process left child 
        if len(left) <= self.min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left, criterion)
            self.split(node['left'], depth+1, criterion)
        # process right child 
        if len(right) <= self.min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right, criterion)
            self.split(node['right'], depth+1, criterion)


    # Build a decision tree
    def fit(self, dataset, criterion):
        # create the root of the tree 
        root = self.get_split(dataset, criterion)

        # split until the tree is formed 
        self.split(root, 1, criterion)
        return root

    # Make a prediction with a decision tree
    def predict(self, node, newData):
        if newData[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], newData)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], newData)
            else:
                return node['right']

    # Predict dataset    
    def predictAll(self, node, X_test):
        predictions = list()
        for row in X_test:
            prediction = self.predict(node, row)
            predictions.append(prediction)
        return(predictions)


    # Print a decision tree
    def print_tree(self, node, depth=0):
        if isinstance(node, dict):
            print('%s[X%d < %.3f]' % ((depth*' ', (node['index']+1), node['value'])))
            self.print_tree(node['left'], depth+1)
            self.print_tree(node['right'], depth+1)
        else:
            print('%s[%s]' % ((depth*' ', node)))
