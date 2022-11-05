from bisect import bisect_left
import numpy as np
import scipy
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram
from matplotlib import pyplot as plt

class KeyWrapper:
    def __init__(self, iterable, key):
        self.it = iterable
        self.key = key

    def __getitem__(self, i):
        return self.key(self.it[i])

    def __len__(self):
        return len(self.it)

class Cluster():

    def __init__(self,id,elems,sick_quantity,centroid=True):
        self.id = id
        self.elems = elems
        self.N = len(elems)
        self.sick_quantity = sick_quantity
        self.healthy_quantity = self.N - self.sick_quantity

        self.centroid = np.average(elems,axis=0) if (centroid and (self.N > 0)) else None
    def __eq__(self, other: object) -> bool:
            #print("id: ",self.id, " other: ",other.id)
            return isinstance(other, Cluster) and self.id == other.id

    def __hash__(self):
            return hash(self.id)


    
    def __repr__(self) -> str:
        return f"Cluster {self.id} ->({self.N})"

    @staticmethod
    def centroid_distance(c1,c2):
        # print(c1)
        # print(c2)
        c1_centroid = np.average(c1.elems,axis=0)
        c2_centroid = np.average(c2.elems,axis=0)
        # c1_centroid = c1.centroid
        # c2_centroid = c2.centroid
        return np.linalg.norm(c2_centroid-c1_centroid,ord=2)

    @staticmethod
    def average_distance(c1,c2):
        pass

    @staticmethod
    def maximum_distance(c1,c2):
        pass

    @staticmethod
    def minimum_distance(c1,c2):
        pass


        


class ClusterHierarchy():

    def __init__(self,df):
        self.is_sick_attr = 'sigdz'
       
        self.df = df
        
        print("len original = ",len(self.df.index))
        self.scaler = StandardScaler()
    
        self.df = self.df[self.df['tvdlm'].notnull()]
        self.df = self.df[self.df['choleste'].notnull()]
        self.df = self.df.reset_index()
        print("df not nul: \n",self.df)
        self.expected_outputs = self.df.loc[:,self.is_sick_attr]
        self.df = self.df.loc[:,['age','cad.dur','choleste']]
        print("Sick quantity: " ,len(self.expected_outputs[self.expected_outputs == 1].index))
        print("Healthy quantity: " ,len(self.expected_outputs[self.expected_outputs == 0].index))
        self.N = len(self.df.index)
        print("Len without Nans: ",self.N)
        self.distance_method = None
        self.sick_clusters = dict()
        self.clusters = None
        
        #print("Clusters = \n",self.clusters)

        #print("Distances = \n",self.distances)
        
    
  
        

    def init_groups(self,centroid=False):
        
        elems = self.df.to_numpy().reshape(-1,3)
     
        elems = self.scaler.fit_transform(elems)
        # Z  =scipy.cluster.hierarchy.linkage(elems, method='centroid', metric='euclidean', optimal_ordering=False)
        # dendrogram (Z = Z,p=5,truncate_mode='level',count_sort='ascending')
        # plt.show()
        #print(elems)
        clusters = list()
        for i in range(self.N):
            is_sick = self.expected_outputs[i] == 1
            cluster = Cluster(id = i,elems = np.array([elems[i]]),sick_quantity=1 if is_sick  else 0,centroid=centroid)
            clusters.append(cluster)
            self.sick_clusters[cluster] = is_sick
        print("init groups end")
        return clusters

    def minimum_idx( arr):
        # Devuelve los n indices mas grandes de mas grande a mas chico como si fuese un array 1D
        idx = np.argpartition(arr, 1, axis=None)
        # Convierto los indices 1D en indices de un array con shape arr.shape
        return np.unravel_index(idx[0], arr.shape)

    def build_groups(self,distance_method='centroid'):
        centroid = distance_method == 'centroid'
        self.clusters = self.init_groups(centroid)
        self.distance_method = self.get_distance_method(distance_method)
    
        groups_quantity = len(self.df.index)
        last_idx = self.N-1

        linkages = []
        self.distances = self.calculate_distances(self.clusters,self.N)
        print("calculate distances end")
        self.removed_clusters = dict()
        self.removed_clusters[None] =True
       
        while groups_quantity > 1:
            print("group quantity = ",groups_quantity)
            #Consigo clusters c1 y c2 con distancia minima
            c1 = None
            c2 = None
            while(self.removed_clusters.get(c1) is not None or self.removed_clusters.get(c2) is not None):
                min_distance_data = self.distances.pop(0)
                #print("min distance data = ",min_distance_data)
                c1 = min_distance_data[1][0]
                c2 = min_distance_data[1][1]
        

            # print("c1: ",c1)
            # print("c2: ",c2)
            self.removed_clusters[c1] = c1
            self.removed_clusters[c2] = c2
            self.sick_clusters[c1] =c1.sick_quantity >= c1.healthy_quantity
            self.sick_clusters[c2] =c2.sick_quantity >= c2.healthy_quantity
            min_distance = min_distance_data[0]
           
            # #Borro de la lista de clusters
            self.clusters.remove(c1)
            self.clusters.remove(c2)
            
            last_idx+=1
            new_cluster = Cluster(id = last_idx,elems = np.append(c1.elems,c2.elems,axis=0),sick_quantity=c1.sick_quantity + c2.sick_quantity,centroid = centroid)
            #Agrego el nuevo cluster
            self.clusters.append(new_cluster) 
            self.sick_clusters[new_cluster] = new_cluster.sick_quantity >= new_cluster.healthy_quantity

            #Disminuyo en 1 la cantidad de grupos
            groups_quantity-=1

           
            #Calculo las distancias al nuevo cluster
            new_distances = self.update_distances(self.clusters,groups_quantity,new_cluster)
            #self.distances = [d for d in self.distances if d[1][0] != c1 and d[1][0] != c2 and d[1][1] != c1 and d[1][1] != c2]
            self.distances.extend(new_distances)
            self.distances = sorted(self.distances, key=lambda x: x[0])
            
            print("update distances end")
      
            #Agrego linkage para sklearn
            linkages.append([c1.id,c2.id,min_distance,new_cluster.N])
           
        #Grafico dendograma
        #print(self.sick_clusters)
        self.create_dendrogram(linkages,5,None)


    def get_color(self,cluster_id):
        more_sick_than_healthy = self.sick_clusters.get(Cluster(cluster_id,[],0))
        if more_sick_than_healthy is None:
            print(f"cluster_id: {cluster_id} is None")
        if more_sick_than_healthy:
            #print("more than healthy: ",more_sick_than_healthy)
            return "#FF0000"
        else:
            #print("more than healthy: ",more_sick_than_healthy)
            return "#00D253"

      

    def create_dendrogram(self,Z,p,truncate_mode=None):
        dendrogram (Z = Z,p=p,truncate_mode=truncate_mode,count_sort='ascending',link_color_func=self.get_color)
        plt.show()
        
    

    
    def get_distance_method(self,distance_method):
        if distance_method == "centroid":
            return Cluster.centroid_distance
        elif distance_method == "maximum":
            pass
        elif distance_method == "minimum":
            pass
        elif distance_method == "average":
            pass
    


    def calculate_distances(self,clusters,N):
        distances = list()
        for i in range(N):
           
            for j in range(i+1,N):
                c1 = clusters[i]
                c2 = clusters[j]
                dist = self.distance_method(c1,c2)
                dist_data = (dist,(c1,c2))
                #print(f"i:{i}, j: {j}) dist = {dist}")
                distances.append(dist_data)
                #print(distances)
        distances.sort(key = lambda x: x[0])
        return distances

    def update_distances(self,clusters,N,new_cluster):
        new_distances = list()
        #Hasta N-1 porque el nuevo cluster está al final siempre
        for i in range(N-1):
        
            #Calculo distancia al nuevo cluster
            dist_to_new_cluster = self.distance_method(clusters[i],new_cluster)
            #print(f"i:{i}, j: {N-1}) dist_to_new_cluster = {dist_to_new_cluster}")

            #Inserto distancia al nuevo cluster en lista distancias
            # bslindex = bisect_left(KeyWrapper(self.distances, key=lambda c: c[0]), dist_to_new_cluster)
            # print(f"{i}) index = ",bslindex)
            dist_data = (dist_to_new_cluster,(clusters[i],new_cluster))
            new_distances.append(dist_data)
        return new_distances



                    

