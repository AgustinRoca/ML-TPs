from pickle import FALSE
from bayes.naive_bayes import NaiveBayes
class Ej1:


    def __init__(self,df):
        self.df = df

        self.naive_bayes = NaiveBayes(self.df, {'scones': [0, 1], 'cerveza': [
                                0, 1], 'wiskey': [0, 1], 'avena': [0, 1], 'futbol': [0, 1], 'Nacionalidad': ['E', 'F']}, 'Nacionalidad')
    def run(self):
        # Ej b)
        new_instance = {'scones': 1, 'cerveza': 0,
                        'wiskey': 1, 'avena': 1, 'futbol': 0}
        opt_class, probability = self.naive_bayes.get_optimum_class_probability(
            new_instance,False)
        print(
            f"b) New instance: {new_instance}\nOptimum class found was {opt_class} with probability {probability}")

        # Ej c)
        new_instance = {'scones': 0, 'cerveza': 1,
                        'wiskey': 1, 'avena': 0, 'futbol': 1}
        opt_class, probability = self.naive_bayes.get_optimum_class_probability(
            new_instance,False)
        print(
            f"c) New instance: {new_instance}\nOptimum class found was {opt_class} with probability {probability}")
