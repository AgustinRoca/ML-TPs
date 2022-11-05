import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class Kohonen():
    def __init__(self, min_vals, max_vals, map_side = 10, initial_learning_factor=0.1, initial_radius = 10):
        self.learning_factor = initial_learning_factor
        self.initial_learning_factor = initial_learning_factor
        self.radius = initial_radius
        self.initial_radius = initial_radius
        self.map_side = map_side
        self.map = []
        self.activation_map = []
        self.winner_distances_history = []
        self.reset_weights(min_vals, max_vals)

    def reset_weights(self, min_vals, max_vals):
        self.map = []
        self.activation_map = []
        min_vals = np.array(min_vals)
        max_vals = np.array(max_vals)
        for i in range(self.map_side):
            self.map.append([])
            self.activation_map.append([])
            for j in range(self.map_side):
                self.map[i].append(np.random.random(3)*(max_vals - min_vals) + min_vals)
                self.activation_map[i].append(0)
        self.map = np.array(self.map)
        self.activation_map = np.array(self.activation_map)

    def run(self, df: pd.DataFrame, epochs = 500):
        for epoch in range(epochs):
            print(f'Epoch: {epoch} - Radius: {self.radius} - η: {self.learning_factor}')
            self.winner_distances_history.append([])
            for row in df.iterrows():
                row_df = row[1]
                entry = np.array((row_df['age'], row_df['cad.dur'], row_df['choleste']))
                winner_i, winner_j = self.get_winner_neuron(entry, epoch)
                self.activation_map[winner_i][winner_j] += 1
                self.map[winner_i][winner_j] -= self.learning_factor * (self.map[winner_i][winner_j] - entry)
                rows, columns = np.indices((len(self.map), len(self.map[0])))
                indices = np.stack((rows, columns), axis=-1)
                distances = np.linalg.norm(indices - np.array([winner_i, winner_j]), axis=2)
                neighbors_is, neighbors_js = np.where(distances <= self.radius)
                for i,j in zip(neighbors_is, neighbors_js):
                    if((i,j) != (winner_i,winner_j)):
                        neuron_distance = distances[i][j]
                        self.map[i][j]-= self.learning_factor * np.exp(-2 * neuron_distance/self.radius) * (self.map[i][j] - entry)

            self.learning_factor = self.initial_learning_factor * (1 - (epoch+1)/epochs)
            self.radius = (epochs - (epoch+1)) * self.initial_radius /epochs
        self.winner_distances_history = np.array(self.winner_distances_history)

    def get_winner_neuron(self, entry, epoch):
        distances = np.linalg.norm(self.map - entry, axis=2) 
        if epoch is not None: 
            self.winner_distances_history[epoch].append(np.amin(distances))
        i,j = np.where(distances == np.amin(distances))
        return i[0], j[0]

    def get_umatrix(self):
        umatrix = []
        for i in range(self.map_side):
            umatrix.append([])
            for j in range(self.map_side):
                umatrix[i].append(self.avg_neighbor_distance(i, j))
        umatrix = np.array(umatrix)
        return umatrix

    def avg_neighbor_distance(self, i, j):
        entry = self.map[i][j]
        min_i = i-1 if i != 0 else 0
        min_j = j-1 if j != 0 else 0
        max_i = i+1 if i != self.map_side - 1 else self.map_side - 1
        max_j = j+1 if j != self.map_side - 1 else self.map_side - 1
        neighbourhood = self.map[min_i: max_i+1, min_j: max_j+1]
        distances = np.linalg.norm(neighbourhood - entry, axis=2)
        return np.sum(distances) / (len(neighbourhood) * len(neighbourhood[0]) - 1)


df_all = pd.read_csv('Data/acath_no_nan.csv', sep=';')
df_men = df_all[df_all['sex'] == 0]
df_women = df_all[df_all['sex'] == 1]
for df in [df_all, df_men, df_women]:
    df = df[['age', 'cad.dur', 'choleste']]
    mean_df = df.mean()
    std_df = df.std()
    df=(df-mean_df)/std_df # normalizo los datos
    mins = np.array([df['age'].min(), df['cad.dur'].min(), df['choleste'].min()])
    maxs = np.array([df['age'].max(), df['cad.dur'].max(), df['choleste'].max()])

    # Corro Kohonen
    kohonen = Kohonen(mins, maxs)
    kohonen.run(df, epochs=500)
    plt.plot(kohonen.winner_distances_history.mean(axis=1))
    plt.show()


    # Hago heatmaps
    sns.set_theme()
    ax = sns.heatmap(kohonen.get_umatrix())
    ax.set_title('U-Matrix')
    plt.show()

    sns.set_theme()
    ax = sns.heatmap(kohonen.activation_map)
    ax.set_title('Activations')
    plt.show()

    # desnormalizo los datos para el heatmap
    result_map = kohonen.map * std_df.to_numpy() + mean_df.to_numpy()

    for i, label in enumerate(['age', 'cad.dur', 'choleste']):
        ax = sns.heatmap(result_map[:,:,i])
        ax.set_title(label)
        plt.show()

    df = pd.read_csv('Data/acath_no_nan.csv', sep=';')[['age', 'cad.dur', 'choleste', 'sigdz']]
    illness_map = []
    for i in range(kohonen.map_side):
        illness_map.append([])
        for j in range(kohonen.map_side):
            illness_map[i].append(0)

    illness_map = np.array(illness_map)
    for x in df.iterrows():
        i, j = kohonen.get_winner_neuron(x[1][['age', 'cad.dur', 'choleste']].to_numpy(), None)
        if x[1]['sigdz'] == 1:
            illness_map[i][j] += 1
        elif x[1]['sigdz'] == 0:
            illness_map[i][j] -= 1

    ax = sns.heatmap(illness_map, annot=True, fmt='d')
    ax.set_title('Illness_map')
    plt.show()

    # sns.heatmap(kohonen.map[:,:,0], annot=True, fmt=".2f")
