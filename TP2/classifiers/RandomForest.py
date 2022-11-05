import numpy as np

from classifiers.ID3Tree import ID3Tree
from classifiers.Classifier import Classifier

class RandomForest(Classifier):
    def __init__(self, attribute, attributes_possible_values, positive = 1, negative = 0, df=None, max_height=10, trees_qty=3):
        self.positive = positive
        self.negative = negative
        self.attribute = attribute
        self.attributes_possible_values = attributes_possible_values
        self.max_height = max_height
        self.trees_qty = trees_qty
        self.trees = []
        if df is not None:
            self.build_trees(df)  

    def train(self, train):
        self.build_trees(train) 
    
    def classify(self, new_instance):
        votes = dict()
        votes[self.positive] = 0
        votes[self.negative] = 0
        for tree in self.trees:
            votes[tree.classify(new_instance)] += 1
        if votes[self.positive] > votes[self.negative]:
            return self.positive
        else:
            return self.negative 

    def get_attribute(self):
        return self.attribute

    def get_possible_values(self):
        return np.array([self.negative, self.positive])

    def build_trees(self, df):
        self.trees = []
        for i in range(self.trees_qty):
            train_df = self._get_bootstrap_df(df)
            tree = ID3Tree(self.attribute, self.attributes_possible_values, self.positive, self.negative, train_df, self.max_height)
            self.trees.append(tree)

    def _get_bootstrap_df(self, train_df):
        train_indexes = np.random.default_rng().choice(train_df.index, len(train_df), replace=True)
        train_df =  train_df.loc[train_indexes]
        return train_df

    def get_avg_nodes(self):
        accum = 0
        for tree in self.trees:
            accum += tree.node_quantity
        return accum / len(self.trees)
