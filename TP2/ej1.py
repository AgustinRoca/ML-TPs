from cProfile import label
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from classifiers.ID3Tree import ID3Tree
from classifiers.RandomForest import RandomForest
from utils.CrossValidation import CrossValidation


class Ej1():
    def __init__(self, df, attribute='Creditability'):
        print("Ej 1")

        bins = [0, 4000, 8000, 11000, df['Credit Amount'].max()]
        df['Credit Amount'] = pd.cut(df['Credit Amount'], bins=bins)

        bins = [19, 48, df['Age (years)'].max()]
        df['Age (years)'] = pd.cut(df['Age (years)'], bins=bins)

        bins = [0, 24, df['Duration of Credit (month)'].max()]
        df['Duration of Credit (month)'] = pd.cut(df['Duration of Credit (month)'], bins=bins)

        self.df = df
        self.attribute = attribute

        self.attribute_possible_values = dict()
        for column_name in self.df.columns:
            self.attribute_possible_values[column_name] = self.df[column_name].unique()

   
      

        
    def run_CV(self):
 
        tree = ID3Tree(self.attribute, self.attribute_possible_values, max_height=41)
        cross_validation = CrossValidation(None, self.df, self.attribute,np.array([0,1]), tree)
        min_error, best_confusion_matrix, best_test = cross_validation.run()
        test_indexes = best_test.index
        train_indexes = np.delete(np.arange(0, len(self.df.index)), test_indexes)
        best_train = self.df.iloc[train_indexes]
        tree = ID3Tree(self.attribute, self.attribute_possible_values, df=best_train, max_height=41)
        max_height = tree.height
        heights = range(1, max_height + 1, 2)
        print('CV) Max Height possible =', tree.height)
        avg_error_test, confusion_matrix_test = tree.classify_set(best_test)

        print("CV) Confusion matrix: \n",confusion_matrix_test)
        print("CV) Avg Accuracy = ",1-avg_error_test)
        #print("tree: \n",tree.root)
           # ID3
        id3_nodes = []
        id3_train_precisions = []
        id3_test_precisions = []
        for height in heights:
            print('CV) Run with max height =', height)
            tree = ID3Tree(self.attribute,self.attribute_possible_values, max_height=height)
            tree.train(best_train)
            avg_error_train, confusion_matrix_train = tree.classify_set(best_train)
            avg_error_test, confusion_matrix_test = tree.classify_set(best_test)
            id3_nodes.append(tree.node_quantity)
            id3_train_precisions.append(1 - avg_error_train)
            id3_test_precisions.append(1 - avg_error_test)

         # PLOTS
        plt.plot(id3_nodes, id3_train_precisions)
        plt.scatter(id3_nodes, id3_train_precisions, label='Train')
        plt.plot(id3_nodes, id3_test_precisions)
        plt.scatter(id3_nodes, id3_test_precisions, label='Test')
        plt.xlabel('#Nodes')
        plt.ylabel('Precision')
        plt.title(f'Cross Validation with K = {cross_validation.K}')
        plt.legend()
        plt.show()    
        
    def run_id3(self):
        entries_len = len(self.df.index)      
        partition_indexes = np.arange(0, entries_len)
        np.random.shuffle(partition_indexes)
        partition = np.array_split(partition_indexes, 2)
        train = self.df.iloc[partition[0]]
        test =self.df.iloc[partition[1]]
        tree = ID3Tree(self.attribute, self.attribute_possible_values,df=train, max_height=41)
        max_height = tree.height
        heights = range(1, max_height + 1, 2)
        print('ID3) Max Height possible =', tree.height)
        avg_error_test, confusion_matrix_test = tree.classify_set(test)

        print("ID3) Confusion matrix: \n",confusion_matrix_test)
        print("ID3) Avg Accuracy = ",1-avg_error_test)

        # ID3
        id3_nodes = []
        id3_train_precisions = []
        id3_test_precisions = []
        for height in heights:
            print('ID3) Run with max height =', height)
            tree = ID3Tree(self.attribute,self.attribute_possible_values, max_height=height)
            tree.train(train)
            avg_error_train, confusion_matrix_train = tree.classify_set(train)
            avg_error_test, confusion_matrix_test = tree.classify_set(test)
            id3_nodes.append(tree.node_quantity)
            id3_train_precisions.append(1 - avg_error_train)
            id3_test_precisions.append(1 - avg_error_test)

         # PLOTS
        plt.plot(id3_nodes, id3_train_precisions)
        plt.scatter(id3_nodes, id3_train_precisions, label='Train')
        plt.plot(id3_nodes, id3_test_precisions)
        plt.scatter(id3_nodes, id3_test_precisions, label='Test')
        plt.xlabel('#Nodes')
        plt.ylabel('Precision')
        plt.title('ID3')
        plt.legend()
        plt.show()
        return train,test,max_height



    def run_random_forest(self,train,test,max_height):
        heights = range(1,max_height + 1, 2)
     
        forest = RandomForest(self.attribute,self.attribute_possible_values, max_height=max_height)
        forest.train(train)
      
        avg_error_test, confusion_matrix_test = forest.classify_set(test)
      
        print("RANDOM FOREST) Confusion matrix: \n",confusion_matrix_test)
        print("RANDOM FOREST) Avg Accuracy = ",1-avg_error_test)

       # RANDOM FOREST
        rf_nodes = []
        rf_train_precisions = []
        rf_test_precisions = []
        for height in heights:
            print('RANDOM FOREST) Run with max height =', height)
            forest = RandomForest(self.attribute,self.attribute_possible_values, max_height=height,trees_qty=7)
            forest.train(train)
            avg_error_train, confusion_matrix_train = forest.classify_set(train)
            avg_error_test, confusion_matrix_test = forest.classify_set(test)
            rf_nodes.append(forest.get_avg_nodes())
            rf_train_precisions.append(1 - avg_error_train)
            rf_test_precisions.append(1 - avg_error_test)
        
       

        plt.plot(rf_nodes, rf_train_precisions)
        plt.scatter(rf_nodes, rf_train_precisions, label='Train')
        plt.plot(rf_nodes, rf_test_precisions)
        plt.scatter(rf_nodes, rf_test_precisions, label='Test')
        plt.xlabel('#Nodes')
        plt.ylabel('Precision')
        plt.title('Random Forest')
        plt.legend()
        plt.show()

    def plot_attribute(self, attribute='Credit Amount', group_range=1000):
        copy_df = self.df.copy()
        print(copy_df[attribute].max())
        bins = range(copy_df[attribute].min(),copy_df[attribute].max(), group_range)
        labels = range(copy_df[attribute].min(),copy_df[attribute].max()-group_range, group_range)
        print(bins)
        copy_df[attribute] = pd.cut(copy_df[attribute], bins=bins, labels=labels)
        copy_df['Creditability'] = copy_df['Creditability'].apply(lambda x: -1 if x == 0 else 1)
        creditability = copy_df.groupby(attribute)['Creditability'].sum()
        width = group_range - group_range/10
        plt.bar(np.array(creditability.index.values.tolist()) + width/2, creditability.values.tolist(), width=width)
        plt.show()