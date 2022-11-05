
import pandas as pd
import numpy as np
from classifiers.ID3Tree import ID3Tree

df = pd.read_csv("TP2/Data/juega_tenis.csv", sep=",",usecols=["Pronostico","Temperatura","Humedad","Viento","Juega"])


attributes_possible_values = {"Pronostico": ["soleado","nublado","lluvioso"], "Temperatura":["calido","templado","frio"],"Humedad":["alta","normal"],"Viento":["debil","fuerte"],"Juega":["si","no"]}
tree = ID3Tree("Juega",attributes_possible_values,positive = "si",negative = "no",df = df)
print(df)
print(tree.root)
print(f"total nodes: {tree.node_quantity}, height: {tree.height}")

# for i, test in df.iterrows():
#     print(f"{i}) Test instance:\n",test)
#     print("Prediction: ",tree.classify(test))
