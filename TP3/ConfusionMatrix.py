import numpy as np

class ConfusionMatrix():
    def __init__(self, possible_values):
        self.matrix = dict()
        for real_value in possible_values:
            self.matrix[real_value] = dict()
            for predicted_value in possible_values:
                self.matrix[real_value][predicted_value] = 0
        self.possible_values = possible_values
        self.results_added = 0

    def add_result(self, real_value, predicted_value):
        self.matrix[real_value][predicted_value] += 1
        self.results_added += 1

    def calculate_errors(self, method='accuracy'):
        methods = {
            'accuracy': self.calculate_accuracy,
            'precision': self.calculate_precision,
            'recall': self.calculate_recall,
            'F1 score': self.calculate_F1_score,
            'TP rate': self.calculate_TP_rate,
            'FP rate': self.calculate_FP_rate,
        }
        errors = dict()
        for value in self.possible_values:
            TP, TN, FP, FN = self.calculate_category_rates(value)
          
            errors[value] = 1-methods[method](TP, TN, FP, FN)
       
            
        return errors

 

    def calculate_category_rates(self,value):
        TP,TN,FP,FN = 0,0,0,0
        TP = self.matrix[value][value]

        other_values = np.delete(self.possible_values,np.where(np.array(self.possible_values) == value))
  
        for predicted_value in other_values:
            FN += self.matrix[value][predicted_value]
            FP += self.matrix[predicted_value][value]
            for other_value in other_values:
                TN += self.matrix[predicted_value][other_value]
        return TP,TN,FP,FN

    def calculate_metrics(self):
        metrics = dict()

        for value in self.possible_values:
            metrics[value]= dict()
            TP, TN, FP, FN = self.calculate_category_rates(value)            
            metrics[value]["precision"] = self.calculate_precision(TP,FP)
            metrics[value]["accuracy"] = self.calculate_accuracy(TP,TN,FP,FN)
            metrics[value]["f1"] = self.calculate_F1_score(TP,FP,FN)
            metrics[value]["TPrate"] = self.calculate_TP_rate(TP,FP)
            metrics[value]["FPrate"] = self.calculate_FP_rate(TN, FP)
        return metrics

    def calculate_accuracy(self, TP,TN,FP,FN):
        div = (TP+TN+FP+FN)
        if div == 0:
            return 0
        return (TP+TN)/div

    def calculate_precision(self, TP,TN,FP,FN):
        div = (TP+FP)
        if div == 0:
            return 0
        return TP/div

    def calculate_recall(self, TP,TN,FP,FN):
        div = (TP+FN)
        if div == 0:
            return 0
        return TP/div

    def calculate_F1_score(self,TP,TN,FP,FN):
        precision = self.calculate_precision(TP,FP)
        recall = self.calculate_recall(TP,FN)
        div = (precision+recall)
        if div == 0:
            return 0
        return (2*precision*recall)/div

    def calculate_TP_rate(self, TP,TN,FP,FN):
        div = (TP+FP)
        if div == 0:
            return 0
        return TP/div

    def calculate_FP_rate(self, TP,TN,FP,FN):
        div = (FP+TN)
        if div == 0:
            return 0
        return FP/div

    def print_metrics(self):
        metrics = self.calculate_metrics()
        print("###############################################")
        for value in self.possible_values:
            accuracy = metrics[value]["accuracy"]
            precision = metrics[value]["precision"]
            f1_score = metrics[value]["f1"]
            tp_rate = metrics[value]["TPrate"]
            fp_rate = metrics[value]["FPrate"]
            avg_accuracy += accuracy
            avg_precision += precision
            avg_f1_score += f1_score
            avg_tp_rate += tp_rate
            avg_fp_rate += fp_rate
            print("Value: ", value)
            print("Accuracy: ",accuracy)
            print("Precision: ", precision)
            print("F1-Score: ", f1_score)
            print("TP Rate: ", tp_rate)
            print("FP Rate: ", fp_rate)
            print("###############################################")

        total_values = len(self.possible_values)
        print("Average Accuracy: ", avg_accuracy/total_values)
        print("Average Precision: ", avg_precision/total_values)
        print("Average F1-Score: ", avg_f1_score/total_values)
        print("Average TP Rate: ", avg_tp_rate/total_values)
        print("Average FP Rate: ", avg_fp_rate/total_values)
        print("###############################################")

    def __str__(self) -> str:
        # HEADER
        s = '  |'
        for value in self.possible_values:
            s += f" {value} |"
        s += '\n'

        s += '--+'
        for _ in self.possible_values:
            s += '---+'
        s += '\n'

        #ROWS
        for value in self.possible_values:
            s += f'{value} |'
            for predicted_value in self.possible_values:
                s += f' {self.matrix[value][predicted_value]} |'
            s += '\n'
        return s
