veimport numpy.random
import pandas as pd
import plotly
import pickle
import pybnesian as pb
import numpy as np
import math
import sklearn
import pybnesianCPT_to_df
from statistics import mean
import pyarrow
from operator import itemgetter
from EDAspy.optimization.custom import GBN
from sklearn.cluster import KMeans
import matplotlib
from pybnesianCPT_to_df import from_CPT_to_df



np.seterr(divide='ignore', invalid='ignore')


#Cargamos los datos
best = pb.load(r"networks\best_network.pickle")

df=pd.read_csv(r'C:\Users\Victor Alejandre\Desktop\CIG\Implementación\example.csv')
df=df.drop('color',axis=1)
df=df.drop('Unnamed: 0',axis=1)

#Calculo del k-means para comparar

kmeans= KMeans(n_clusters=2,n_init='auto').fit(df)

MAP_kmeans=kmeans.cluster_centers_
print(MAP_kmeans)

#Establecemos los clusters según el k seleccionado:
n_clusters=2
clusters_name=[]
for i in range(1,n_clusters+1):
    clusters_name.append('c'+f'{i}')




def joint_sampling(red,cluster_names,n=100000): #Con esta función obtenemos un sample de tamaño n de la distribución conjunta P(C,X). El sample se realiza en orden ancestral de
    #la red
    ancestral_order= pb.Dag(red.nodes(), red.arcs()).topological_sort()
    samples=pd.DataFrame(columns=ancestral_order)
    for i in range(n):
        evidence={}
        for node in ancestral_order:
            if node=='cluster':
                evidence[node] = numpy.random.choice(cluster_names, 1,
                       p=from_CPT_to_df(str(red.cpd('cluster'))).iloc[0].tolist())[0]

            else:
                parents = pd.DataFrame(columns=list(red.parents(node)))
                for element in red.parents(node):
                    if element == 'cluster':
                        parents[element] = pd.Series(pd.Categorical([itemgetter(element)(evidence)], categories=cluster_names))
                    else:
                        parents[element] = itemgetter(element)(evidence)

                evidence[node] = red.cpd(node).sample(1, parents).tolist()[0]
        x=pd.DataFrame(evidence, index=[0])
        samples=pd.concat([samples,x])
    return samples

#Para el cálculo de representantes realizamos un probabilistic logic sampling. Para ello sampleamos n instancias y para aquellas que tengan cluster c1 la media de dichas instancias
#nos devuelve la media de P(X|C1) y así sucesivamente
def get_MAP(red,clusters_names,n=100000):
    MAP = pd.DataFrame()
    sample=joint_sampling(red,clusters_names,n)

    variables = pb.Dag(red.nodes(), red.arcs()).topological_sort()
    variables.remove('cluster')

    for k in range(len(clusters_names)):
        sample_c=sample.loc[sample['cluster']==clusters_names[k]]
        sample_c=sample_c.drop('cluster',axis=1)
        map=[]
        for i in range(len(variables)):
            var=variables[i]
            x=sample_c[var].mean()
            map.append(x)

        MAP[clusters_names[k]] = map

    MAP = MAP.set_index(np.array(variables))


    plot = MAP.T
    return plot

plot=get_MAP(best,clusters_name)
print(plot)
def hellinger_distance(p, q):
    d = 0
    for i in range(len(p)):
        d = d + (math.sqrt(p[i]) - math.sqrt(q[i])) ** 2
    return math.sqrt(d) / math.sqrt(2)

#Este diccionario contendrá las ft importance para cada representante de cada cluster
importances={}

for clster in plot.index:

    map=plot.loc[clster].tolist()
    #Calculamos la probabilidad a posteriori de C dado el representante
    prob_posterior = []
    lklh = []
    ancestral_order = pb.Dag(best.nodes(), best.arcs()).topological_sort()
    ancestral_order.remove('cluster')

    for c in clusters_name:
        instance = pd.DataFrame([map], columns=ancestral_order)  # map va ordenado en orden ancestral siempre
        instance['cluster'] = pd.Series(pd.Categorical([c], categories=clusters_name))
        x = math.exp(best.logl(instance)[0])

        lklh.append(x)
    t = sum(lklh)
    prob_posterior = [x / t for x in lklh]

    prob_posterior_map = prob_posterior.copy()

    #Comenzamos con el cálculo de las importancias, como tenemos variables continuas vamos a realizar sampling de P(X_i| x_(-i)) de tamaño l
    l = 10000
    ancestral = pb.Dag(best.nodes(), best.arcs()).topological_sort()

    importance = {}

    for k in range(len(ancestral_order)):
        #Sampleamos de toda la distribución de la red y dejamos la variable cluster
        sample_c = best.sample(l).to_pandas()
        sample_c = sample_c.drop('cluster', axis=1)
        #Obtenemos nuestra evidencia que será el valor del MAP para todas las variables menos la de interés
        evidence = {}
        for s in range(len(ancestral_order)):
            if s != k:
                evidence[ancestral_order[s]] = map[s]

        #Sampleamos de P(X_i| x_(-i)) para ello utilizamos edaspy para obtener una muestra de dicha distribución para calcular los parámetros
        red = GBN(sample_c.columns, evidences=evidence)
        red.learn(sample_c)
        sample = red.sample(l)
        sample = pd.DataFrame(sample, columns=sample_c.columns)
        #con la muestra de edaspy podemos aproximar la media y la varianza de P(X_i| x_(-i)) con la cual sampleamos de la normal que teóricamente debería seguir.
        #realizamos este paso debido a que la librería no tiene funcionalidad para acceder a los parámetros
        mean = sample[ancestral_order[k]].mean()
        var = sample[ancestral_order[k]].var()
        sample_var = np.random.normal(mean, var, l)
        #Una vez tenemos una muestra de la distribucion deseada calculamos ft importance por aproximación (realizamos estimador media)
        distances = []
        for i in range(l):

            evidence[ancestral_order[k]] = sample_var[i]
            data = pd.DataFrame(evidence, index=[0])
            lklh = []
            for cluster in clusters_name:
                data['cluster'] = pd.Series(pd.Categorical([cluster], categories=clusters_name))
                x = math.exp(best.logl(data)[0])

                lklh.append(x)

            t = sum(lklh)
            if t != 0:  ## por problemas con pybnesian esta condición se debe introducir.

                prob_posterior = [x / t for x in lklh]


                d = hellinger_distance(prob_posterior_map, prob_posterior)

                distances.append(d)



        importance[ancestral_order[k]] = sum(distances) / len(distances)

    print(importance)
    importances[clster]=importance


print(importances)

