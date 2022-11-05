from cmath import nan
from functools import total_ordering
import logging
import pandas as pd
import sys
from pyparsing import Char
from time import process_time_ns
from ej1 import Ej1
from ej2 import Ej2
import math
import matplotlib.pyplot as plt

def read_data(path, dtypes,usecols):

    return pd.read_excel(path, usecols=usecols, dtype=dtypes)

#################################################################

def ej1(path):
    df = None
    try:
        df = read_data(path, dtypes = {
            'scones': int, 'cerveza': int, 'wiskey': int, 'avena': int, 'futbol': int, 'Nacionalidad': Char}, usecols=['scones', 'cerveza', 'wiskey', 'avena', 'futbol' , 'Nacionalidad'])

    except Exception as e:
        print("An error ocurrer while reading the file")
        logging.exception(e)
        sys.exit(1)
    
    ej1 = Ej1(df)
    ej1.run()
        

#################################################################




def ej2(path):
    df = None
    try:
        start = process_time_ns()
        
        df = read_data(path,dtypes={'titular':str,'categoria':str} ,usecols=['titular','categoria'])
        print(f"Read data elapsed time: {(process_time_ns() - start)/1000000000}s")
    except Exception as e:
        print("An error ocurrer while reading the file")
        logging.exception(e)
        sys.exit(1)
    
    ej2 = Ej2(df)
    ej2.run()


#################################################################

def ej3(path,display_internal_info = False):
    df = pd.read_csv(path)
    tot_row = len(df)
    rank_probabilities = []
    for i in range(1,5):
        rank_probabilities.append(( len(df[df['rank'] == i]) + 1 )/( tot_row + 4 ))
    gre_probabilities = []
    for i in range(1,5):
        gre_probabilities.append([
            (len(df[(df['rank'] == i) & (df['gre'] < 500)])  + 1) / (len(df[df['rank'] == i]) + 2),
            (len(df[(df['rank'] == i) & (df['gre'] >= 500)]) + 1) / (len(df[df['rank'] == i]) + 2)
        ])
    gpa_probabilities = []
    for i in range(1,5):
        gpa_probabilities.append([
            (len(df[(df['rank'] == i) & (df['gpa'] < 3)])  + 1) / (len(df[df['rank'] == i]) + 2),
            (len(df[(df['rank'] == i) & (df['gpa'] >= 3)]) + 1) / (len(df[df['rank'] == i]) + 2)
        ])
    admit_probabilities = []
    for i in range(1,5):
        admit_probabilities.append([])
        for j in range(2):
            admit_probabilities[i-1].append([])
            for k in range(2):
                admit_probabilities[i-1][j].append([])
                for l in range(2):
                    admit_probabilities[i-1][j][k].append(
                        (len(df[
                                (df['rank'] == i) & 
                                (df['gre'] < 500 if j == 0 else df['gre'] >= 500) & 
                                (df['gpa'] < 3 if k == 0 else df['gpa'] >= 3) & 
                                (df['admit'] == l)
                            ])  + 1) 
                        / (len(df[
                                (df['rank'] == i) & 
                                (df['gre'] < 500 if j == 0 else df['gre'] >= 500) & 
                                (df['gpa'] < 3 if k == 0 else df['gpa'] >= 3)
                            ]) + 2)
                    )
    res_q1 = 0
    # P(admit = 0 | gpa >= 3 , gre >= 500, rank = 1) * P(gpa >=3 | rank=1) * P(gre >= 500 | rank = 1) +
    # P(admit = 0 | gpa >= 3 , gre < 500, rank = 1) * P(gpa >=3 | rank=1) * P(gre < 500 | rank = 1) + 
    # P(admit = 0 | gpa < 3 , gre >= 500, rank = 1) * P(gpa < 3 | rank=1) * P(gre >= 500 | rank = 1) + 
    # P(admit = 0 | gpa < 3 , gre < 500, rank = 1) * P(gpa < 3 | rank=1) * P(gre < 500 | rank = 1)
    res_q1 += admit_probabilities[0][1][1][0] * gpa_probabilities[0][1] * gre_probabilities[0][1]
    res_q1 += admit_probabilities[0][0][1][0] * gpa_probabilities[0][1] * gre_probabilities[0][0]
    res_q1 += admit_probabilities[0][1][0][0] * gpa_probabilities[0][0] * gre_probabilities[0][1]
    res_q1 += admit_probabilities[0][0][0][0] * gpa_probabilities[0][0] * gre_probabilities[0][0]
    
    # P(admit = 1 | gpa >= 3, gre < 500, rank = 2)
    res_q2 = 0
    res_q2 += admit_probabilities[1][0][1][1]
    
    print(f'Pregunta 1: {res_q1}')
    print(f'Pregunta 2: {res_q2}')

    if display_internal_info:
        _ej3_display_internal_info(tot_row,rank_probabilities,gre_probabilities,gpa_probabilities,admit_probabilities)

def _ej3_display_internal_info(tot_row,rank_probabilities,gre_probabilities,gpa_probabilities,admit_probabilities):
    print(f'Total rows:{tot_row}')
    rank_table=[
        ['rank','','',''],
        ['1','2','3','4'],
        [str(rank_probabilities[0]),str(rank_probabilities[1]),str(rank_probabilities[2]),str(rank_probabilities[3])]
    ]
    gre_table = [
        ['','gre',''],
        ['rank','< 500','>= 500']
    ]
    for i in range(4):
        gre_table.append([str(i+1),str(gre_probabilities[i][0]),str(gre_probabilities[i][1])])
    gpa_table = [
        ['','gpa',''],
        ['rank','< 3','>= 3'],
    ]
    for i in range(4):
        gpa_table.append([str(i+1),str(gpa_probabilities[i][0]),str(gpa_probabilities[i][1])])
    admit_table = [
        ['','','admit','',''],
        ['rank','gre','gpa','F','T']
    ]
    for i in range(4):
        for j in range(2):
            for k in range(2):
                admit_table.append([
                    str(i+1),
                    '< 500' if j == 0 else '>= 500',
                    '< 3' if k == 0 else '>= 3',
                    str(admit_probabilities[i][j][k][0]),
                    str(admit_probabilities[i][j][k][1]),
                ])
    def _print_table(table):
        longest = 0
        for i in table:
            for j in i:
                if len(j) > longest:
                    longest = len(j)
        print('-'*(longest * len(table[0])+len(table[0])+1))
        for i in table:
            print('|',end='')
            for j in i:
                print(j,end='')
                if len(j) < longest:
                    print(' ' * (longest - len(j)),end='')
                print('|',end='')
            print()
            print('-'*(longest * len(table[0])+len(table[0])+1))
    print()
    _print_table(rank_table)
    print()
    _print_table(gre_table)
    print()
    _print_table(gpa_table)
    print()
    _print_table(admit_table)
    print()


#################################################################
if __name__ == "__main__":

    if len(sys.argv) != 3 and len(sys.argv) != 4:
       print("Uso: python main.py <ejercicio> <data path>")
       sys.exit(1)

    choice = sys.argv[1]
    path = sys.argv[2]
    if choice == "1":
        ej1(path)
    elif choice == "2":
        ej2(path)
    elif choice == "3":
        if len(sys.argv) == 4:
            ej3(path,True)
        else:
            ej3(path)
    else:
        print(f"<ejercicio> must be between 1 and 3")

