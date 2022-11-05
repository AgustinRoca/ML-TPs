from pandas import DataFrame
from utils.Classifier import Classifier
import numpy as np
from scipy import stats


class LinearRegressionML(Classifier):
    def __init__(self, attribute, bins = None):
        self.coef = {}
        self.errors = {}
        self.X = None
        self.Y = None
        self.y_attribute = attribute
        self.bins = bins
    
    def fit(self,X:DataFrame,Y, verbose=False):
        self.X = X.to_numpy().T[0]
        self.Y = Y
        y_mean = self.Y.mean()
        x_mean = self.X.mean()
        numerator = 0
        denominator = 0
        for x,y in zip(self.X,self.Y):
            numerator += (x-x_mean) * (y-y_mean)
            denominator += (x-x_mean)**2
        self.coef = [0, 0]
        self.coef[1] = numerator/denominator
        self.coef[0] = y_mean - self.coef[1] * x_mean

        sumatoria = 0
        for y in self.Y:
            sumatoria += (y - y_mean)**2
        rse = np.sqrt(sumatoria/(len(Y) - 2))
        self.errors = [0, 0]
        self.errors[0] = np.sqrt(rse**2 * (1/len(self.X) + x_mean**2 /denominator))
        self.errors[1] = np.sqrt(rse**2 / denominator)

        if verbose:
            print(f"B0 = {self.coef[0]}; error = {self.errors[0]}")
            print(f"B1 = {self.coef[1]}; error = {self.errors[1]}")
            print()

    def pvalue(self):
        t = self.coef[1]/self.errors[1]
        pvalue = stats.t.sf(np.abs(t), self.X.shape[0]-2)*2
        print(f"p-value: {pvalue}")
        
    def r2(self):
        y_mean = self.Y.mean()
        Y_adj = self.coef[1] * self.X + self.coef[0]
        rss=0
        tss=0
        for y, y_adj in zip(self.Y, Y_adj):
            rss += (y-y_adj)**2
            tss += (y-y_mean)**2
        r2 = 1 - rss/tss
        print(f"r2: {r2}")

    def mse(self):
        Y_adj = self.coef[1] * self.X + self.coef[0]
        rss=0
        for y, y_adj in zip(self.Y, Y_adj):
            rss += (y-y_adj)**2
        mse = (1/self.X.shape[0]) * rss
        print(f"MSE: {mse}")

    def mae(self):
        Y_adj = self.coef[1] * self.X + self.coef[0]
        rss=0
        for y, y_adj in zip(self.Y, Y_adj):
            rss += np.abs(y-y_adj)
        mae = (1/self.X.shape[0]) * rss
        print(f"MAE: {mae}")

    def rss(self):
        Y_adj = self.coef[1] * self.X + self.coef[0]
        rss=0
        for y, y_adj in zip(self.Y, Y_adj):
            rss += (y-y_adj)**2
        print(f"RSS: {rss}")

    def train(self, train):
        X = train.drop(['Sales'], axis=1)
        Y = train["Sales"]
        self.fit(X, Y)

    def classify(self, new_instance):
        y = self.coef[1] * new_instance.to_numpy().T[0] + self.coef[0]
        for bin in self.bins:
            if y in bin:
                return bin

    def get_attribute(self):
        return self.y_attribute

    def get_possible_values(self):
        return self.bins
