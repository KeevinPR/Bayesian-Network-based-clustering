import pandas as pd
import pybnesian as pb
import itertools
import math
import numpy as np




dataframe=pd.read_csv(r'C:\Users\G513\PycharmProjects\cluster_intrepretability\datasets\customers.csv')

#Preprocesado de los datos donde convertimos en categórica la variables continuas income y age
dataframe=dataframe.drop('ID',axis=1)
dataframe = dataframe.sample(n=200, random_state=42).reset_index(drop=True)

def ni(x):
    if x<50000:
        return 'Low income'
    elif 52200<=x<130000:
        return 'Middle income'
    else:
        return 'High income'

def na(x):
    if x<26:
        return 'Junior'
    elif 26<=x<55:
        return 'Adult'
    else:
        return 'Senior citizen'


new_income=[]
new_age=[]
for i in range(dataframe.shape[0]):
    new_income.append(ni(dataframe['Income'][i]))
    new_age.append(na(dataframe['Age'][i]))
dataframe['Income']=new_income
dataframe['Age']=new_age

#Cambio del formato de las variables int a variables tipo string para poder trabajar con ellas
for var in dataframe.columns:
    dataframe[var]=dataframe[var].apply(lambda x: str(x))

#Base network (Naive Bayes)
#Construimos una red base Naive Bayes con la variable cluster con el k seleccionado y las variables de estudio.
#structure
#Creamos los arcos para ello para cada variable del dataframe creamos el arco (cluster,variable)
n_clusters=2
in_arcs=[]
for var in dataframe.columns:
    in_arcs.append(('cluster',var))
#Creamos los nodos, para ello comenzamos introduciendo el nodo cluster y tras ello para cada variable introducimos un nodo
in_nodes=['cluster']
for var in dataframe.columns:
    in_nodes.append(var)
#El siguiente código convierte cada variable del dataframe a tipo categórico para poder trabajar con las librerias
for var in dataframe.columns:
    dataframe[var]=dataframe[var].astype('category')
#Finalmente creamos los clusters a partir de el k establecido. Esto se realiza para luego poder desarrollar el código.
clusters=[]
for i in range(1,n_clusters+1):
    clusters.append('c'+f'{i}')
#Finalmente creamos una red discreta con pybnesian con los nodos y arcos generados
red_inicial=pb.DiscreteBN(in_nodes,in_arcs)



#Aprendizaje del modelo
from discrete_structure import sem

#Necesitamos los posibles valores de cada variable para trabajar con los algoritmos implementados, estos deben ir en un diccionaro {variable: [categorias]}
if __name__ == '__main__':
    for var in dataframe.columns:
        dataframe[var]=dataframe[var].astype('category')

    categories={}
    for var in dataframe.columns:
        categories[var]=dataframe[var].cat.categories


    best=sem(red_inicial,dataframe,categories,clusters)

    print(best.arcs())



    best.save("best_network_customers_2_parallelized", include_cpd= True)
