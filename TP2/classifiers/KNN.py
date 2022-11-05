import numpy as np

from classifiers.Classifier import Classifier


class KNN(Classifier):

    def __init__(self,K,category_label,possible_values):
        self.K = K
        self.category_label = category_label
        self.possible_values = possible_values

    def train(self, train):
        self.train_set = train

    def classify(self, test_instance):
        closests= self.get_closests(test_instance)
        
        classified  = False
        current_K = self.K
        while not classified:
            value_counter = dict()
            closest_K = closests.head(current_K)
            for index in closest_K.index:
                value = self.train_set.loc[index][self.category_label]
                if value in value_counter:
                    value_counter[value] += 1
                else:
                    value_counter[value] = 1
            max_class_list = self.get_max_class_list(value_counter)
            if len(max_class_list) == 1:
                return max_class_list[0]
            else:
               
             
                if current_K < len(closests):
                    current_K +=1
                else:
                    return  max_class_list[0]

    def get_max_class_list(self, dictionary):
        max_value = max(dictionary.values())
        return [k for k,v in dictionary.items() if v == max_value]

    def calculate_distance(self, x1, x2):
        x1 = x1.to_numpy()
        x2 = x2.to_numpy()
        return np.linalg.norm(x1-x2)
    
    def get_closests(self, test_instance):
        distances = self.get_distances(test_instance)
       
        return distances.sort_values()

    def get_distances(self, test_instance):
        column_names = self.train_set.columns.values.tolist()
        column_names.remove(self.category_label)
        distances = self.train_set.copy()
        return distances[column_names].apply(lambda row: self.calculate_distance(test_instance,row), axis=1)

    def get_attribute(self):
        return self.category_label

    def get_possible_values(self):
        return self.possible_values