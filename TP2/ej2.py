import numpy as np
from classifiers.KNN import KNN
from classifiers.WKNN import WKNN

from utils.CrossValidation import CrossValidation
import matplotlib.pyplot as plt
class Ej2():

    def __init__(self,df):
        print("Ej 2")
        self.df = df
        self.data_columns = ["wordcount","titleSentiment","sentimentValue"]
        self.category_label = "Star Rating"
        self.category_possible_values = np.arange(1,6) # de 1 estrella a 5 estrellas

        self.df["titleSentiment"] = df["titleSentiment"].map({"positive":1,"negative":0})
        
        self.df[self.data_columns] = self.normalize(self.df, self.data_columns)
        self.complete_missing_data()

    def complete_missing_data(self):
        category_label = self.data_columns[1]
        wknn = WKNN(1,  category_label, np.array([0,1]))
        neighbours = self.df[self.df[category_label].notnull()]
        #print("neighbours data: \n",neighbours)
        incomplete_data_set  = self.df[self.df[category_label].isnull()]
        #print("incomplete data: \n",incomplete_data_set)
        wknn.train(neighbours)
        for i,incomplete_data in incomplete_data_set.iterrows():
            #print("incomplete data: \n",incomplete_data)
            incomplete_data  = incomplete_data.drop(category_label)
            closest_index = wknn.get_closests(incomplete_data).head(1).index
            
            self.df.at[i,category_label] = self.df.loc[closest_index][category_label]
            #print("New data: \n",self.df.loc[i])
    def plot_data(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')

        # For each set of style and range settings, plot n random points in the box
        # defined by x in [23, 32], y in [0, 100], z in [zlow, zhigh].
        for category in self.category_possible_values:
            xs = self.df[self.df[self.category_label] == category][self.data_columns[0]]
            ys = self.df[self.df[self.category_label] == category][self.data_columns[2]]
            zs = self.df[self.df[self.category_label] == category][self.data_columns[1]]
            ax.scatter(xs, ys, zs,label=f"Star Rating = {category}")

        ax.set_xlabel(self.data_columns[0])
        ax.set_ylabel(self.data_columns[2])
        ax.set_zlabel(self.data_columns[1])
        ax.legend()
        plt.show()

    def normalize(self,df,columns):
        return (df[columns] - df[columns].mean()) / df[columns].std()  

    def run(self, cross_validation_K=10):
        knn_errors = []
        wknn_errors = []
        Ks = range(1, len(self.df.index) - int(np.floor(len(self.df.index)/cross_validation_K)), 10)
        for K in Ks:
            error_knn, error_wknn = 0, 0
            iterations = 1
            for i in range(iterations):
                print('Running', i + 1, 'of', iterations, 'iterations with K=', K)

                # Weighted-KNN
                wknn = WKNN(K, self.category_label, self.category_possible_values)
                cross_validation_wknn = CrossValidation(cross_validation_K, self.df, self.category_label, self.category_possible_values, wknn, error_method='precision')
                new_error_wknn, matrix_wknn, test_wknn = cross_validation_wknn.run()
                error_wknn += new_error_wknn

                # KNN
                knn = KNN(K, self.category_label, self.category_possible_values)
                cross_validation_knn = CrossValidation(cross_validation_K, self.df, self.category_label, self.category_possible_values, knn, test=test_wknn, error_method='precision')
                new_error_knn, matrix_knn, test_knn = cross_validation_knn.run()
                error_knn += new_error_knn            
            
            knn_errors.append(error_knn/iterations)
            wknn_errors.append(error_wknn/iterations)
            
            print('--------------KNN--------------')
            print(matrix_knn)
            print('Average error KNN:', error_knn/iterations)
            print('--------------WKNN--------------')
            print(matrix_wknn)
            print('Average error WKNN:', error_wknn/iterations)
        
        plt.plot(Ks, knn_errors)
        plt.scatter(Ks, knn_errors, label='KNN')
        plt.plot(Ks, wknn_errors)
        plt.scatter(Ks, wknn_errors, label='WKNN')

        plt.xlabel('K')
        plt.ylabel('Error')
        plt.legend()
        plt.show()
    