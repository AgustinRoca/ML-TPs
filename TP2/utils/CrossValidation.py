import numpy as np
from utils.ConfusionMatrix import ConfusionMatrix
from classifiers.Classifier import Classifier
import matplotlib.pyplot as plt
class CrossValidation():
    """
    Performs a cross validation given a K, a pandas dataframe, the title of the category to cross validate, its possible values
    """
    def __init__(self,K, df, category_label, category_values, classifier : Classifier, test=None, error_method='accuracy'):
        self.K = K if test is None else 1
        self.df = df
        self.test = test
        self.category_label = category_label
        self.category_values = category_values
        self.classifier = classifier
        self.MAX_K = 30
        self.error_method = error_method

    def _get_partition_indexes(self, test,K):
        if test is not None:
            return [test.index]
        else:
            entries_len = len(self.df.index)      
            partition_indexes = np.arange(0, entries_len)
            np.random.shuffle(partition_indexes)
        
            return np.array_split(partition_indexes, K)

    def _get_test_and_train_dfs(self, i):
        test_indexes = self.partition_indexes[i]
        test = self.df.iloc[test_indexes]
        train_indexes = np.delete(np.arange(0, len(self.df.index)), test_indexes)
        train = self.df.iloc[train_indexes]
        return test, train

    def _cross_validate(self,k):
        min_error = None
        best_confusion_matrix = None
        best_train = None
        best_test = None
        errors = list()
       
        for i in range(k):
            test, train = self._get_test_and_train_dfs(i)
            self.classifier.train(train)
        
            confusion_matrix = ConfusionMatrix(self.category_values)
            for _,test_instance in test.iterrows():
                real_class = test_instance[self.category_label]
                test_instance = test_instance.drop(self.category_label)
                predicted_class = self.classifier.classify(test_instance)
                confusion_matrix.add_result(real_class, predicted_class)
    
            error = np.mean(list(confusion_matrix.calculate_errors(self.error_method).values()))
            errors.append(error)
            if (not min_error) or (error < min_error):
                min_error = error
                best_train = train
                best_test = test
                best_confusion_matrix = confusion_matrix
        return min_error,np.mean(errors),best_confusion_matrix,best_train,best_test

    def run(self):

        #print(f"Running Cross Validation with K = {self.K}")
        min_error = None
        best_confusion_matrix = None
        best_train = None
        best_test = None
        min_K = 2
        if self.K is None:
            min_errors = list()
            avg_errors = list()
            Ks = range(min_K,self.MAX_K,4)
            for k in Ks:
             
                self.partition_indexes = self._get_partition_indexes(self.test,k)
                min_err,avg_err,confusion_matrix,train,test = self._cross_validate(k)
                min_errors.append(min_err)
                avg_errors.append(avg_err)
                print(f"K = {k}, Min Error = {min_err}, Avg Error = {avg_err}")
                if (not min_error) or (min_err < min_error):
                 
                    min_K = k
                    self.K = min_K
                    min_error = min_err
                    best_train = train
                    best_test = test
                    best_confusion_matrix = confusion_matrix
                print(f"Min error for K = {min_K}")
            plt.plot(Ks,min_errors,'o',label="Minimum Error")
            plt.plot(Ks,avg_errors,'o',label="Average Error")
            plt.xlabel("Cross Validation K")
            plt.ylabel("Error")
            plt.legend()
            plt.show()

        else:
             self.partition_indexes = self._get_partition_indexes(self.test,self.K)
             min_error,avg_error,best_confusion_matrix,best_train,best_test = self._cross_validate(self.K)
            # print("Error:", min_error)
            # print(best_confusion_matrix)
        
        return min_error, best_confusion_matrix, best_test
