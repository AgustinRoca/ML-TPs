import math
from traceback import print_tb
import pandas as pd
import numpy as np
from classifiers.Classifier import Classifier
import sys
class ID3Tree(Classifier):
    def __init__(self, attribute, attributes_possible_values ,positive = 1, negative = 0, df=None, max_height=10):
        self.positive = positive
        self.negative = negative
        self.attribute = attribute
        self.attributes_possible_values = attributes_possible_values
        self.root = None
        self.height = 0
        self.node_quantity = 0
        self.max_height = max_height
        if df is not None:
            self.root,self.node_quantity,self.height = self.build_tree(df, attribute)   

    def build_tree(self, df, parent_df=None, current_height=1):
        # Casos base
        if df.empty:
            return self.ID3Node(parent_df[self.attribute].mode()[0] == self.positive),1,0
        if (df[self.attribute] == self.positive).all():
            return self.ID3Node(True),1,1
        if (df[self.attribute] == self.negative).all():
            return self.ID3Node(False),1,1
        attributes = df.drop(self.attribute, axis=1)
        if attributes.empty or current_height == self.max_height:
            return self.ID3Node(df[self.attribute].mode()[0] == self.positive),1,1
        
        # Elijo raiz
        root = self.ID3Node(self.get_highest_gain_attribute(df, self.attribute))
        total_node_quantity = 1

        # Genero recursivamente los nodos hijos
        max_subtree_height = 0
        children_values = self.attributes_possible_values[root.value]
        for child_value in children_values:
            child_df = df[df[root.value]==child_value].drop(root.value, axis=1)
            child, node_quantity, subtree_height = self.build_tree(child_df, df, current_height+2)
            if subtree_height > max_subtree_height:
                max_subtree_height = subtree_height
            total_node_quantity += node_quantity + 1
            root.add_child(child_value,child) 
        return root, total_node_quantity, max_subtree_height + 2

    def train(self, df):
        self.root,self.node_quantity,self.height = self.build_tree(df)
  
    def classify(self,instance):
        if self.root is None:
            return None
        node = self.root
        while not node.isLeaf:
            try:
                instance_value = instance[node.value]
                node = node.children[instance_value]
            except:
                print("An exception occurred")
                print(f"node: {node.isLeaf} , {node}")
                print("instance: \n",instance)
             
                sys.exit(1)
          
        return self.positive if node.value else self.negative

    def get_attribute(self):
        return self.attribute

    def get_possible_values(self):
        return np.array([self.negative, self.positive])

    def get_highest_gain_attribute(self, df, attribute):
        gains = self.calculate_gains(df, attribute)      
        highest = gains.sort_values('gain', ascending=False)['column_name'].values[0]
        return highest

    def calculate_gains(self, df, attribute):
        column_names = df.columns.values.tolist()
        column_names.remove(self.attribute)
      
        gains = []
        df_entropy = self.calculate_entropy(df, attribute)
        for i, column_name in enumerate(column_names):
            gains.append([column_name, df_entropy])
            relative_frecs = df[column_name].value_counts(normalize=True)
            for value in relative_frecs.index:
                gains[i][1] = gains[i][1] - relative_frecs.loc[value] * self.calculate_entropy(df[df[column_name] == value], attribute)
        gains = np.array(gains, ndmin=2)
     
        gains_df = pd.DataFrame(data=gains, columns=['column_name', 'gain'])
        gains_df = gains_df.astype({'column_name': str, 'gain': float})

        return gains_df
                
    def calculate_entropy(self, df, attribute):
        entropy = 0
        relative_frecs = df[attribute].value_counts(normalize=True)
        for value in relative_frecs.index:
            p = relative_frecs.loc[value]
            entropy -= p * math.log(p, 2)
        return entropy

    class ID3Node:
        def __init__(self, val):
            self.value = val
            self.isLeaf = type(val) is bool or type(val) is np.bool_          
            self.children  = None

        def add_child(self,child_value,node):
            if child_value is None or node is None:
                return
            if self.children is None:
                self.children = dict()
            self.children[child_value] = node

        def __str__(self) -> str:
            if self.isLeaf:
                return f"({str(self.value)})"
            s = f"({self.value}"
            if self.children is None:
                s += ")"
            else:
                key_len = len(self.children.keys())
            
                for i,key in enumerate(self.children.keys()):
                    if i < key_len-1:

                        s+= f"({key}{str(self.children[key])}, "
                    else:
                        s+= f"{key}{str(self.children[key])})"
            return s
        
