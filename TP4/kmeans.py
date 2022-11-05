from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from utils.Classifier import Classifier

class KMeans(Classifier):
    def __init__(self ):
        pass

    def _compare_kgroup(self, kgroup1,kgroup2):
        return np.count_nonzero(np.subtract(kgroup1,kgroup2))/len(kgroup1)
    
    def _calculate_centroids(self,df:pd.DataFrame):
        k_groups = df['k_group'].unique()
        centroids = np.zeros((len(k_groups),df.shape[1]))
        for i in range(len(k_groups)):
            centroids[i] = np.append(np.average(df[df['k_group'] == k_groups[i]].drop('k_group',axis=1),axis=0),k_groups[i])
        return centroids

    def _calculate_k_group(self,df:pd.DataFrame,centroids:np.ndarray):
        dists = np.empty((2,df.shape[0]))
        dists[0] = df.sub(centroids[0,:-1], axis=1).pow(2).sum(axis=1).pow(.5)
        dists[1] = centroids[0,-1]
        for i in range(1, centroids.shape[0]):
            new_dists = df.sub(centroids[i,:-1], axis=1).pow(2).sum(axis=1).pow(.5)
            ind = new_dists<dists[0]
            dists[0][ind] = new_dists[ind]
            dists[1][ind] = centroids[i,-1]
        return dists[1,:]

    def fit(self,df:pd.DataFrame,k = 2,max_iter = 1000):
        if k < 1:
            raise Exception("k must be at least 1")
        df = df.copy()
        df['k_group'] = np.random.randint(0, k, df.shape[0])
        k_group_prev = df["k_group"]
        self.centroids = self._calculate_centroids(df)
        df['k_group'] = self._calculate_k_group(df.drop('k_group',axis=1),self.centroids)
        i = 1 
        while i < max_iter and self._compare_kgroup(df['k_group'],k_group_prev) != 0:
            i += 1
            k_group_prev = df['k_group']
            self.centroids = self._calculate_centroids(df)
            df['k_group'] = self._calculate_k_group(df.drop('k_group',axis=1),self.centroids)
        return df

    def predict(self,val):
        return self._calculate_k_group(val,self.centroids)

    def plot(self, df:pd.DataFrame):
        df = df.copy()
        colors = plt.cm.get_cmap('rainbow',len(self.centroids))
        for i in range(len(self.centroids)):
            currgroup = df[df['k_group'] == self.centroids[i,-1]]
            plt.scatter(currgroup['age'],currgroup['choleste'],color=colors(i))
            plt.scatter(self.centroids[i][0],self.centroids[i][1], color='black',marker='x')
            plt.annotate(f'{int(self.centroids[i,-1])}', (self.centroids[i][0],self.centroids[i][1]))
        plt.gca().set_aspect('equal', adjustable='box')
        plt.show()
    
    def get_average_distances(self, df):
        # dists = np.empty((2,df.shape[0]))
        # dists[0] = df.sub(centroids[0,:-1], axis=1).pow(2).sum(axis=1).pow(.5)
        # dists[1] = centroids[0,-1]
        # for i in range(1, centroids.shape[0]):
        #     new_dists = df.sub(centroids[i,:-1], axis=1).pow(2).sum(axis=1).pow(.5)
        #     ind = new_dists<dists[0]
        #     dists[0][ind] = new_dists[ind]
        #     dists[1][ind] = centroids[i,-1]
        # return dists[1,:]
        avgdists = []
        k_groups = df['k_group'].unique()
        for i in range(len(k_groups)):
            centroid = self.centroids[self.centroids[:,-1] == k_groups[i],:-1]
            auxdf = df[df['k_group'] == k_groups[i]].drop('k_group',axis=1)
            avgdists.append(
                np.average(
                    auxdf.sub(centroid, axis=1).pow(2).sum(axis=1).pow(.5)
                )
            )
        return np.average(avgdists)


    def train(self, train):
        df = train.copy()
        mean_df = df.mean()
        std_df = df.std()
        normdf=(df-mean_df)/std_df

        self.only_two = False
        if self.only_two:
            grouped = self.fit(normdf[['age','choleste']],5)
        else:
            grouped = self.fit(normdf[['age','choleste','cad.dur']],5)

        df['k_group'] = grouped['k_group']
        self.classification = {}
        for g in df['k_group'].unique():
            count_0 = len(df[(df['k_group'] == g) & (df['sigdz'] == 0) ])
            count_1 = len(df[(df['k_group'] == g) & (df['sigdz'] == 1) ])
            self.classification[g] = 0 if count_0 > count_1 else 1

    def classify(self, new_instance):
        new_instance = pd.DataFrame([new_instance.to_numpy()], columns=['sex', 'age', 'cad.dur', 'choleste', 'tvdlm'])
        if self.only_two:
            return self.classification[self.predict(new_instance[['age','choleste']])[0]]
        else:
            return self.classification[self.predict(new_instance[['age','choleste','cad.dur']])[0]]

    def get_attribute(self):
        return 'sigdz'

    def get_possible_values(self):
        return {self.classification.values}
