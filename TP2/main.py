
import logging
import pandas as pd
import sys

from time import process_time_ns
from ej1 import Ej1
from ej2 import Ej2



def read_data(path, dtypes, usecols, sep=';'):

    return pd.read_csv(path,sep=sep, usecols=usecols, dtype=dtypes)

#################################################################

def ej1(path):
    df = None
    try:
        df = read_data(path, 
        dtypes = {
            'Creditability': int,
            'Account Balance': int,
            'Duration of Credit (month)': int,
            'Payment Status of Previous Credit': int,
            'Purpose': int,
            'Credit Amount': int,
            'Value Savings/Stocks': int,
            'Length of current employment': int,
            'Instalment per cent': int,
            'Sex & Marital Status': int,
            'Guarantors': int,
            'Duration in Current address': int,
            'Most valuable available asset': int,
            'Age (years)': int,
            'Concurrent Credits': int,
            'Type of apartment': int,
            'No of Credits at this Bank': int,
            'Occupation': int,
            'No of dependents': int,
            'Telephone': int,
            'Foreign Worker': int},
        usecols=[
            'Creditability',
            'Account Balance',
            'Duration of Credit (month)',
            'Payment Status of Previous Credit',
            'Purpose',
            'Credit Amount',
            'Value Savings/Stocks',
            'Length of current employment',
            'Instalment per cent',
            'Sex & Marital Status',
            'Guarantors',
            'Duration in Current address',
            'Most valuable available asset',
            'Age (years)',
            'Concurrent Credits',
            'Type of apartment',
            'No of Credits at this Bank',
            'Occupation',
            'No of dependents',
            'Telephone',
            'Foreign Worker'],
        sep=',')
    except Exception as e:

        print("An error ocurrer while reading the file")
        logging.exception(e)
        sys.exit(1)
    
    ej1 = Ej1(df)
    train,test, max_height = ej1.run_id3()
    ej1.run_random_forest(train,test,max_height)
    ej1.run_CV()

    #ej1.run()
        

#################################################################




def ej2(path):
    df = None
    try:
        start = process_time_ns()
        
        df = read_data(path,usecols=['wordcount','titleSentiment','sentimentValue','Star Rating'],dtypes={'wordcount':int,'titleSentiment':str,'sentimentValue':float,'Star Rating':int})
        print(f"Read data elapsed time: {(process_time_ns() - start)/1000000000}s")
        print(f"Total data: {df.shape[0]}")
        print('-------------------------------')
        print('Ej 2.a')
        print('Word count promedio en 1*:', df[(df['Star Rating'] == 1)]['wordcount'].mean())
        print('-------------------------------')
    except Exception as e:
        print("An error ocurrer while reading the file")
        logging.exception(e)
        sys.exit(1)
    
    ej2 = Ej2(df)
    #ej2.plot_data()
    #ej2.analyze_data()
  
    
    #ej2.plot_data()
    #ej2.plot_data(False)
    #ej2.cross_validation_K_analysis()
    
    ej2.run(run_not_normalized=True)


#################################################################




#################################################################
if __name__ == "__main__":

    if len(sys.argv) != 3:
       print("Uso: python main.py <ejercicio> <data path>")
       sys.exit(1)

    choice = sys.argv[1]
    path = sys.argv[2]
    if choice == "1":
        ej1(path)
    elif choice == "2":
        ej2(path)
    else:
        print("Uso: python main.py <ejercicio> <data path>")

