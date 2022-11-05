from pandas import DataFrame
import numpy as np
from scipy import stats


class MultipleLinearRegressionML:
    def __init__(self, attribute, bins = None):
        self.coef = []
        self.X = None
        self.Y = None
        self.y_attribute = attribute
        self.bins = bins
    
    def fit(self,X:DataFrame,Y, verbose=False):
        X.insert(loc=0, column='1s', value=1)
        self.X = X
        self.Y = Y
        self.coef = np.linalg.pinv(self.X.T.dot(self.X)).dot(self.X.T).dot(self.Y)
        if verbose:
            print(f"B_0 = {self.coef[0]}; B_1 = {self.coef[1]}; B_2 = {self.coef[2]}; B_3 = {self.coef[3]}")

    def r2(self):
        Y_adj = self.X.dot(self.coef)
        y_mean = self.Y.mean()
        rss = 0
        tss = 0
        for y, y_adj in zip(self.Y, Y_adj):
            rss += (y-y_adj)**2
            tss += (y-y_mean)**2
        std_squared = rss / (self.X.shape[0]-self.X.shape[1]-1)
        variance = std_squared * np.linalg.pinv(self.X.T.dot(self.X))
        r2 = 1 - rss/tss
        print(f"R2 = {r2}")

    def r2_adj(self):
        Y_adj = self.X.dot(self.coef)
        y_mean = self.Y.mean()
        rss = 0
        tss = 0
        for y, y_adj in zip(self.Y, Y_adj):
            rss += (y-y_adj)**2
            tss += (y-y_mean)**2
        std_squared = rss / (self.X.shape[0]-self.X.shape[1]-1)
        variance = std_squared * np.linalg.pinv(self.X.T.dot(self.X))
        r2 = 1 - rss/tss
        r2_adj = 1 - ((1-r2)*(self.X.shape[0]-1)/(self.X.shape[0] - self.X.shape[1]))
        print(f"R2_adj = {r2_adj}")

    def pvalue(self):
        Y_adj = self.X.dot(self.coef)
        y_mean = self.Y.mean()
        rss = 0
        tss = 0
        for y, y_adj in zip(self.Y, Y_adj):
            rss += (y-y_adj)**2
            tss += (y-y_mean)**2
            
        f = ((tss - rss) / self.X.shape[1]) / (rss / (self.X.shape[0] - self.X.shape[1] - 1))
        print(f"f: {f}")

        pvalue = 1-stats.f.cdf(f, self.X.shape[0] - 1, self.X.shape[1])
        print(f"p-value: {pvalue}")

    def rss(self):
        Y_adj = self.X.dot(self.coef)
        rss = 0
        for y, y_adj in zip(self.Y, Y_adj):
            rss += (y-y_adj)**2
        print(f"RSS: {rss}")

    def train(self, train):
        X = train.drop(['Sales'], axis=1)
        Y = train["Sales"]
        self.fit(X, Y)

    def classify(self, new_instance):
        y = self.coef[0] + self.coef[1:].dot(new_instance)
        for bin in self.bins:
            if y in bin:
                return bin

    def get_attribute(self):
        return self.y_attribute

    def get_possible_values(self):
        return self.bins