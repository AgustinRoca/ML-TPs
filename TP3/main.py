
import logging
import pandas as pd
import sys

from time import process_time_ns
from ej1 import Ej1
from ej2 import Ej2



def read_data(path, dtypes, usecols, sep=';'):

    return pd.read_csv(path,sep=sep, usecols=usecols, dtype=dtypes)

#################################################################

def ej1():
    ej1 = Ej1()
    ej1.run()

#################################################################

def ej2():
    ej2 = Ej2()
    ej2.run()

#################################################################

if __name__ == "__main__":

    if len(sys.argv) != 2:
       print("Uso: python main.py <ejercicio>")
       sys.exit(1)

    choice = sys.argv[1]
  
    if choice == "1":
        ej1()
    elif choice == "2":
        ej2()
    else:
        print("Uso: python main.py <ejercicio>")

