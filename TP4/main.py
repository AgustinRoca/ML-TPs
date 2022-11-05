from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import sys
from utils.CrossValidation import CrossValidation
from clusterHierarchy import ClusterHierarchy
from logisticRegression import LogisticRegression
from kmeans import KMeans




#################################################################

def ej1():
    pass

#################################################################

def ej2():
    pass

#################################################################

def ejkmeans(df:pd.DataFrame,k=5):
    df = df.copy()
    only_two = True
    if only_two:
        mean_df = df[['age','choleste']].mean()
        std_df = df[['age','choleste']].std()
    else:
        mean_df = df[['age','choleste','cad.dur']].mean()
        std_df = df[['age','choleste','cad.dur']].std()
    normdf=(df-mean_df)/std_df
    normdf['sigdz'] = df['sigdz']

    kmeans = KMeans()
    if only_two:
        grouped = kmeans.fit(normdf[['age','choleste']],k)
        kmeans.plot(grouped)
    else:
        grouped = kmeans.fit(normdf[['age','choleste','cad.dur']],k)

    normdf['k_group'] = grouped['k_group']
    classification = {}
    for g in normdf['k_group'].unique():
        count_0 = len(normdf[(normdf['k_group'] == g) & (normdf['sigdz'] == 0) ])
        count_1 = len(normdf[(normdf['k_group'] == g) & (normdf['sigdz'] == 1) ])
        print(f'{count_0} y {count_1} para {g}')
        classification[g] = 0 if count_0 > count_1 else 1
    print(classification)


def ejkmeans_crossvalidation(df:pd.DataFrame):
    cross = CrossValidation(None,df,'sigdz',[0,1],KMeans())
    err,mat,test = cross.run()
    print(f'err:{err}, mat:{mat},test:{test}')

    
    

def ejkmeans_find_k(df:pd.DataFrame):
    df = df.copy()
    only_two = True
    if only_two:
        mean_df = df[['age','choleste']].mean()
        std_df = df[['age','choleste']].std()
        normdf=(df[['age','choleste']]-mean_df)/std_df
    else:
        mean_df = df[['age','choleste','cad.dur']].mean()
        std_df = df[['age','choleste','cad.dur']].std()
        normdf=(df[['age','choleste','cad.dur']]-mean_df)/std_df

    avgdists = []

    K_RANGE = range(1,10)
    TEST_TRIES = 10
    for k in K_RANGE:
        avgdist = []
        for i in range(TEST_TRIES):
            kmeans = KMeans()
            grouped = kmeans.fit(normdf,k)
            normdf['k_group'] = grouped['k_group']
            aux_avg = kmeans.get_average_distances(normdf)
            avgdist.append(aux_avg)
        avgdists.append(np.average(avgdist))
    plt.scatter(K_RANGE,avgdists)
    plt.plot(K_RANGE,avgdists)
    plt.show()


def sex_histogrmas(df,sex,title):
    data = df[df['sex']== sex]['sigdz']

    plt.hist(data)
    plt.title(title)
    plt.show()

if __name__ == "__main__":

    if len(sys.argv) != 3:
       print("Uso: python main.py <ejercicio> <path>")
       sys.exit(1)

    choice = sys.argv[1]
    path= sys.argv[2]
    if choice == "1":
        df = pd.read_csv(path, 
                          
                        usecols=[
                            'sex',
                            'age',
                            'cad.dur',
                            'choleste',
                            'sigdz',
                            'tvdlm'],
                        sep=';')
        print(df)
        sex_histogrmas(df,0,"Sigdz en sexo masculino")
        sex_histogrmas(df,1,"Sigdz en sexo femenino")
        logit = LogisticRegression(df)
        logit.ejB()
    elif choice == "3":
        df = pd.read_csv(path, 
                          
                        usecols=[
                            'sex',
                            'age',
                            'cad.dur',
                            'choleste',
                            'sigdz',
                            'tvdlm'],
                        sep=';')
        print(df)
        
        
        clusters_hierarchy = ClusterHierarchy(df)
        clusters_hierarchy.build_groups()
        

    elif choice == '2':
        df = pd.read_csv(path, 
                          
                        usecols=[
                            'sex',
                            'age',
                            'cad.dur',
                            'choleste',
                            'sigdz',
                            'tvdlm'],
                        sep=';')
        ejkmeans_find_k(df)
        ejkmeans_crossvalidation(df)
        ejkmeans(df)
    elif choice == '2-0':
        # Masculino
        df = pd.read_csv(path, 
                          
                        usecols=[
                            'sex',
                            'age',
                            'cad.dur',
                            'choleste',
                            'sigdz',
                            'tvdlm'],
                        sep=';')
        df = df[df['sex'] == 0]
        ejkmeans_find_k(df)
        ejkmeans_crossvalidation(df)
        ejkmeans(df,4)
    elif choice == '2-1':
        # Femenino
        df = pd.read_csv(path, 
                          
                        usecols=[
                            'sex',
                            'age',
                            'cad.dur',
                            'choleste',
                            'sigdz',
                            'tvdlm'],
                        sep=';')
        df = df[df['sex'] == 1]
        ejkmeans_find_k(df)
        ejkmeans_crossvalidation(df)
        ejkmeans(df)

    else:
        print("Uso: python main.py <ejercicio>")

