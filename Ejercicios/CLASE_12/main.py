import sys
from matplotlib import pyplot as plt
import pandas as pd
from utils.CrossValidation import CrossValidation
import numpy as np
from sklearn.metrics import accuracy_score
from itertools import chain


from linear_regression import LinearRegressionML
from multiple_linear_regression import MultipleLinearRegressionML


def ej1(df):
    # Get data
    print('Datos')
    print(df.head())
    print('Covarianza')
    print(df.cov())
    print('Correlation')
    print(df.corr())

    # Plot
    plt.scatter(df["TV"],df["Sales"],label="TV")
    plt.scatter(df["Radio"],df["Sales"],label="Radio")
    plt.scatter(df["Newspaper"],df["Sales"],label="Newspaper")
    
    plt.legend()
    plt.xlabel('TV/Radio/Newspaper')
    plt.ylabel('Sales')

    plt.show()

def ej2(df):
    for ad_type in df[["TV","Radio","Newspaper"]]:
        lr = LinearRegressionML('Sales')
        lr.fit(df[[ad_type]],df["Sales"])
        plt.scatter(df[ad_type],df["Sales"])
        x = np.linspace(df[ad_type].min(),df[ad_type].max(),100)
        y = lr.coef[1]*x+lr.coef[0]
        plt.plot(x, y, '-r')
        plt.title(f"RLS of {ad_type} and Sales")
        plt.xlabel(ad_type)
        plt.ylabel("Sales")
        plt.show()

def ej3(df):
    mlr = MultipleLinearRegressionML('Sales')
    mlr.fit(df[["TV","Radio","Newspaper"]],df["Sales"])

def ej4(df):
    X = df[["TV","Radio","Newspaper"]]
    for ad_type in X:
        x = X[[ad_type]]
        Y = df["Sales"]
        lr = LinearRegressionML('Sales')
        lr.fit(x,Y, verbose=True)
        print()
        lr.pvalue()
        print()
        lr.r2()
        print()
        lr.mse()
        print()
        lr.mae()
        print()
        lr.rss()
        print()
    print("----------------------------------------------------------")
    mlr = MultipleLinearRegressionML('Sales')
    mlr.fit(X, Y, verbose=True)
    print()
    mlr.r2()
    print()
    mlr.r2_adj()
    print()
    mlr.pvalue()
    print()
    mlr.rss()

def ej5():
    print("El ej5 es analizar la influencia de cada variable en las ventas. Esto se puede ver mirando los coeficientes de cada modelo")

def ej6(df):
    bins = [0, 5, 10, 15, 20, df['Sales'].max()]
    discrete_df = df.copy()
    discrete_df['Sales'] = pd.cut(discrete_df['Sales'], bins=bins)
    unique_lists_in_items = discrete_df['Sales'].unique().tolist()
    for ad_type in df[["TV", "Radio", "Newspaper"]]:
        print(ad_type)
        lr = LinearRegressionML('Sales', unique_lists_in_items)
        cross_validation = CrossValidation(10, df[[ad_type, 'Sales']], 'Sales', unique_lists_in_items, lr)
        min_error, best_confusion_matrix, best_test = cross_validation.run()
        metrics = best_confusion_matrix.calculate_metrics()
        print(best_confusion_matrix)
        print(min_error)
        for value in unique_lists_in_items:
            print()
            print(value)
            print(f"Accuracy: {metrics[value]['accuracy']}")
            print(f"Precision: {metrics[value]['precision']}")
            print(f"Recall: {metrics[value]['recall']}")

    print("----------------------------------------------------------")
    mlr = MultipleLinearRegressionML('Sales', unique_lists_in_items)
    cross_validation_mlr = CrossValidation(10, df[["TV","Radio","Newspaper","Sales"]], 'Sales', unique_lists_in_items, mlr)
    min_error_mlr, best_confusion_matrix_mlr, best_test = cross_validation_mlr.run()
    metrics_mlr = best_confusion_matrix_mlr.calculate_metrics()
    print(best_confusion_matrix_mlr)
    print(min_error_mlr)
    for value in unique_lists_in_items:
        print()
        print(value)
        print(f"Accuracy: {metrics_mlr[value]['accuracy']}")
        print(f"Precision: {metrics_mlr[value]['precision']}")
        print(f"Recall: {metrics_mlr[value]['recall']}")


if __name__ == "__main__":
    if len(sys.argv) > 2:
        df = pd.read_csv(sys.argv[1])
        
        if sys.argv[2] == '1':
            ej1(df)
        elif sys.argv[2] == '2':
            ej2(df)
        elif sys.argv[2] == '3':
            ej3(df)
        elif sys.argv[2] == '4':
            ej4(df)
        elif sys.argv[2] == '5':
            ej5()
        elif sys.argv[2] == '6':
            ej6(df)