import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import sys

GRASAS_SAT = 0
ALCOHOL = 1
CALORIAS = 2
SEXO = 3
df = None
headers = None


def filterColumn(col):
    return np.extract(col != 999.99, col)


##################### Ej 1 #####################

def ej1(df):
    grasas_sat = filterColumn(df[headers[GRASAS_SAT]].to_numpy())
    alcohol = filterColumn(df[headers[ALCOHOL]].to_numpy())
    calorias = filterColumn(df[headers[CALORIAS]].to_numpy())

    fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(16, 6))
    fig.suptitle('Ejercicio 1', fontsize=16)
    gs_bplot = ax1.boxplot(grasas_sat, patch_artist=True)
    gs_bplot['boxes'][0].set_facecolor('pink')
    ax1.set(ylabel="Grasas Saturadas")

    al_bplot = ax2.boxplot(alcohol, patch_artist=True)
    al_bplot['boxes'][0].set_facecolor('lightblue')
    ax2.set(ylabel="Alcohol")

    cal_bplot = ax3.boxplot(calorias, patch_artist=True)
    cal_bplot['boxes'][0].set_facecolor('lightgreen')
    ax3.set(ylabel="Calorías")

    print("Media de grasas saturadas totales: ",
          round(np.mean(grasas_sat), 2))
    print("Mediana de grasas saturadas totales: ",
          round(np.median(grasas_sat), 2))
    print("")

    print("Media de alcohol total: ",
          round(np.mean(alcohol), 2))
    print("Mediana de alcohol total: ",
          round(np.median(alcohol), 2))
    print("")

    print("Media de calorías totales: ",
          round(np.mean(calorias), 2))
    print("Mediana de calorías totales: ",
          round(np.median(calorias), 2))
    print("")

    plt.show()
    


##################### Ej 2 #####################


def printStats(stat, M_data, F_data):
    print("\n#########################################################\n")
    print("Media de ", stat, " en sexo masculino: ",
          round(np.mean(M_data), 2))
    print("Mediana de ", stat, " en sexo masculino: ",
          round(np.median(M_data), 2))
    print("")
    print("Media de ", stat, " en sexo femenino: ",
          round(np.mean(F_data), 2))
    print("Mediana de ", stat, " en sexo femenino: ",
          round(np.median(F_data), 2))


def make_stat_graphics(stat_index, stat_name, M_rows, F_rows, bins):
    PADDING_MULTIPLIER = 1/10

    # Stats
    M_stat = filterColumn(M_rows[headers[stat_index]].to_numpy())
    F_stat = filterColumn(F_rows[headers[stat_index]].to_numpy())

    # Ranges
    stat_range = (min(M_stat.min(),F_stat.min()), max(M_stat.max(),F_stat.max()))
    plot_padding = (stat_range[1] - stat_range[0]) * PADDING_MULTIPLIER

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.suptitle(f"{stat_name} p/sexo", fontsize=16)

    # Boxplot masculino
    ax1 = axes[0, 0]
    M_bplot = ax1.boxplot(M_stat, patch_artist=True)
    M_bplot['boxes'][0].set_facecolor('lightblue')
    ax1.set(ylabel=stat_name)
    # ax1.set_title("Sexo Masculino")
    ax1.set_ylim((stat_range[0] - plot_padding,stat_range[1] + plot_padding))
    
    # Histograma masculino
    ax2 = axes[0, 1]
    hist1 = ax2.hist(M_stat, bins,range=(stat_range[0],stat_range[1]), color='lightblue')
    ax2.set(xlabel=stat_name)
    ax2.set_xlim((stat_range[0] - plot_padding,stat_range[1] + plot_padding))
    
    # Boxplot femenino
    ax3 = axes[1, 0]
    F_bplot = ax3.boxplot(F_stat, patch_artist=True)
    F_bplot['boxes'][0].set_facecolor('pink')
    ax3.set(ylabel=stat_name)
    # ax3.set_title("Sexo Femenino")
    ax3.set_ylim((stat_range[0] - plot_padding,stat_range[1] + plot_padding))

    # Histograma femenino
    ax4 = axes[1, 1]
    hist2 = ax4.hist(F_stat, bins,range=(stat_range[0],stat_range[1]), color='pink')
    ax4.set(xlabel=stat_name)
    ax4.set_xlim((stat_range[0] - plot_padding,stat_range[1] + plot_padding))
    
    # Set histogram y_lim
    hist_range = (min(min(hist1[0]),min(hist2[0])), max(max(hist1[0]),max(hist2[0])))
    hist_padding = (hist_range[1] - hist_range[0]) * PADDING_MULTIPLIER

    ax2.set_ylim((0, hist_range[1] + hist_padding))
    ax4.set_ylim((0, hist_range[1] + hist_padding))

    printStats(stat_name, M_stat, F_stat)
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in ['lightblue','lightpink']]
    labels = ["Sexo Masculino", "Sexo Femenino"]
    fig.legend(handles, labels, loc='upper right')  
   
    #plt.legend(handles, labels)
    plt.show()


def ej2(df):
    M_rows = df[df[headers[SEXO]] == "M"]
    F_rows = df[df[headers[SEXO]] == "F"]

    print("Cantidad de Sexo Masculino = ", len(M_rows))
    print("Cantidad de Sexo Femenino = ", len(F_rows))

    n_bins = 30
    # Grasas saturadas p/sexo
    make_stat_graphics(GRASAS_SAT, "Grasas saturadas", M_rows, F_rows, n_bins)

    # Alcohol p/sexo
    make_stat_graphics(ALCOHOL, "Alcohol", M_rows, F_rows, n_bins)

    # Calorías p/sexo
    make_stat_graphics(CALORIAS, "Calorías", M_rows, F_rows, n_bins)


def ej3(df):
    categ_1 = filterColumn(df[(df[headers[CALORIAS]] >= 1) & (
        df[headers[CALORIAS]] <= 1100)][headers[ALCOHOL]].to_numpy())  # [1,1100]
    
    categ_2 = filterColumn(df[(df[headers[CALORIAS]] >
                  1100) & (df[headers[CALORIAS]] <= 1700)][headers[ALCOHOL]].to_numpy())  # (1100,1700]
    categ_3 = filterColumn(df[df[headers[CALORIAS]]
                 > 1700][headers[ALCOHOL]].to_numpy())  # (1700, infinito)

    print("\n#########################################################\n")
    print(f"Categ 1: {len(categ_1)} personas")
    alcohol_med_1 = np.average(categ_1)
    print("Alcohol promedio de Categoría 1 = ", round(alcohol_med_1,2))

    print(f"Categ 1: {len(categ_2)} personas")
    alcohol_med_2 = np.average(categ_2)
    print("Alcohol promedio de Categoría 2 = ", round(alcohol_med_2,2))

    print(f"Categ 1: {len(categ_3)} personas")
    alcohol_med_3 = np.average(categ_3)
    print("Alcohol promedio de Categoría 3 = ", round(alcohol_med_3,2))

    N, bins, patches = plt.hist(df[headers[CALORIAS]], edgecolor='black', bins=[
                             0, 1100, 1700, np.max(df[headers[CALORIAS]].to_numpy())])
    plt.ylabel("Personas")
    colors = ['yellow','orange','red']
    CATEG_1 = 0
    CATEG_2 = 1
    CATEG_3 = 2
    patches[CATEG_1].set_facecolor(colors[CATEG_1])
    patches[CATEG_2].set_facecolor(colors[CATEG_2])
    patches[CATEG_3].set_facecolor(colors[CATEG_3])
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors]
    labels= ["CATEG 1","CATEG 2", "CATEG 3"]
    plt.suptitle('Cantidad de personas por categoría', fontsize=16)
    plt.legend(handles, labels)
    plt.show()

    data = [categ_1,categ_2,categ_3]
    plt.suptitle('Alcohol por categoría', fontsize=16)
    bplot = plt.boxplot(data, patch_artist=True)
    bplot['boxes'][CATEG_1].set_facecolor(colors[CATEG_1])
    bplot['boxes'][CATEG_2].set_facecolor(colors[CATEG_2])
    bplot['boxes'][CATEG_3].set_facecolor(colors[CATEG_3])
    handles = [Rectangle((0, 0), 1, 1, color=c, ec="k") for c in colors]
    labels = ["CATEG 1", "CATEG 2", "CATEG 3"]
    plt.ylabel("Alcohol")
    plt.legend(handles, labels)
  

    plt.show()

    xy  = df[(df[headers[ALCOHOL]] != 999.99) & (df[headers[CALORIAS]] != 999.99)]
    x = xy[headers[CALORIAS]].to_numpy()
    y = xy[headers[ALCOHOL]].to_numpy()
    
    plt.plot(x,y,'o')
    plt.suptitle('Dispersión Calorías / Alcohol', fontsize=16)
    plt.xlabel("Calorías")
    plt.ylabel("Alcohol")
   
    plt.show()
   


if __name__ == "__main__":

    try:
        df = pd.read_excel('./Ejercicios/CLASE_1/Data/Datos trabajo 1.xls', dtype={
            'Grasas_sat': float, 'Alcohol': float, 'Calorías': float, 'Sexo': str})
        headers = df.columns.values

    except Exception as e:
        print("An error ocurrer while reading the file")
        logging.exception(e)
        sys.exit(1)

    if len(sys.argv) == 1:
        ej1(df)
        ej2(df)
        ej3(df)
    else:
        choice = sys.argv[1]
        if choice == "ej1":
            ej1(df)
        elif choice == "ej2":
            ej2(df)
        else:
            ej3(df)
