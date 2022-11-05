from abc import ABC, abstractmethod
from utils.ConfusionMatrix import ConfusionMatrix
import numpy as np

class Classifier(ABC):

    @abstractmethod
    def train(self, train):
        pass

    @abstractmethod
    def classify(self, new_instance):
        pass

    @abstractmethod
    def get_attribute(self):
        pass

    @abstractmethod
    def get_possible_values(self):
        pass

    def classify_set(self, test, error_method='accuracy'):
        confusion_matrix = ConfusionMatrix(self.get_possible_values())
        for _,test_instance in test.iterrows():
            real_class = test_instance[self.get_attribute()]
            test_instance = test_instance.drop(self.get_attribute())
            predicted_class = self.classify(test_instance)
            confusion_matrix.add_result(real_class, predicted_class)
        
        return np.mean(list(confusion_matrix.calculate_errors().values())), confusion_matrix