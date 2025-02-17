import pandas as pd
import plotly
import pickle
import pybnesian as pb
import numpy as np
import math
import sklearn
import pybnesianCPT_to_df
from operator import itemgetter
from radar_chart_discrete import ComplexRadar
import radar_chart_discrete_categories
import math
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
from statistics import mean
import itertools
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter
import numpy
from sklearn import preprocessing
from pybnesianCPT_to_df import from_CPT_to_df




#def joint_sampling(red,cluster_names,n=100000)
def joint_sampling(red,cluster_names,n=100):  #esta funcion realiza un sampling de la Red Bayesiana de forma que obtenemos una muestra de la distribución conjunta de tamaño n
    ancestral_order = pb.Dag(red.nodes(), red.arcs()).topological_sort()
    samples = pd.DataFrame(columns=ancestral_order) #creamos el dataframe donde vamos a guardar los samples
    for i in range(n):
        evidence = {}
        for node in ancestral_order:
            if node == 'cluster': #sampleamos de la variable cluster
                evidence[node] = numpy.random.choice(cluster_names, 1,
                                                     p=from_CPT_to_df(str(red.cpd('cluster'))).iloc[0].tolist())[0]

            else: #sampleamos cada variable dado el valor obtenido de sus padres, de ahí el uso de ancestral order
                prob = from_CPT_to_df(str(red.cpd(node)))
                for element in red.parents(node): #vamos reduciendo el dataframe hasta obtener una única fila que contendra P(X| Pa(X))
                    prob=prob.loc[prob[element]==itemgetter(element)(evidence)]
                    prob=prob.drop(element,axis=1)
                cat = prob.columns #definimos las posibles categorías que toma la variable
                prob=prob.iloc[0].tolist()
                prob=[float(x) for x in prob]
                t=sum(prob) #normalizamos las probabilidades debido a que Pybnesian puede devolver probabildiades que suman un valor muy cercano a uno pero no uno y esto da error
                evidence[node] = numpy.random.choice(cat, 1,
                                                     p=[x/t for x in prob])[0] #sampleamos el valor de la variable dada las probabilidades
        x = pd.DataFrame(evidence, index=[0])
        samples = pd.concat([samples, x]) #añadimos el sample al dataframe
    return samples

def get_MAP(red,clusters_names,n=200000): #Esta función nos devuelve el map dado cada cluster, es decir, el representante de cada cluster. La inferencia se realiza
    #mediante probabilistic logic sampling. Para ello se toma una muestra de la distribución conjunta (joint_sampling) y nos quedamos con los samples que necesitamos.
    MAP = pd.DataFrame()
    sample=joint_sampling(red,clusters_names,n) #obtenemos una muestra de la distribución conjunta

    variables = pb.Dag(red.nodes(), red.arcs()).topological_sort()
    variables.remove('cluster')
    for k in range(len(clusters_names)): #para cada cluster obtenemos el MAP
        sample_c=sample.loc[sample['cluster']==clusters_names[k]] #nos quedamos con los samples cuyo valor es el del cluster de estudio
        sample_c=sample_c.drop('cluster',axis=1)
        dict=sample_c.value_counts().to_dict()
        map=list(max(dict, key=dict.get)) #obtenemos el MAP
        for i in range(len(variables)): #Calculamos la probabilidad para cada variable de P(valor MAP variable | C) que será mostrado en el radar chart
            var=variables[i]
            p=sample_c.loc[sample_c[var]==map[i]][var].tolist().count(map[i])/sample_c.shape[0]
            map[i]=(map[i],p)

        MAP[clusters_names[k]] = map

    MAP = MAP.set_index(np.array(variables))
    print(MAP)

    plot = MAP.T
    return plot




def naming(dataframe_map, importance,red): #Esta función nos devuelve el Radar Chart con los representantes de cada cluster, su importancia y las probabilidades condicionadas de cada
    #valor del representante dado el cluster
    bounds = []
    for var in dataframe_map.columns:
        bounds.append([0, 1])
    min_max_per_variable = pd.DataFrame(bounds, columns=['min', 'max'])
    variables = dataframe_map.columns
    ranges = list(min_max_per_variable.itertuples(index=False, name=None))

    fig1 = plt.figure(figsize=(10, 10))
    radar = ComplexRadar(fig1, variables, ranges, show_scales=True)


    for g in dataframe_map.index:
        info = f"cluster {g}"
        for var in importance[g].keys():
            info = info + f"\n {var, dataframe_map.loc[g][var][0]} importance {round(importance[g][var],4)}"
        radar.plot(dataframe_map.loc[g].values, label=info, )
        radar.fill(dataframe_map.loc[g].values, alpha=0.5)

    radar.set_title("MAP representative of each cluster")
    # radar.use_legend(loc='lower left', bbox_to_anchor=(0, -0.55,1,0.75),ncol=radar.plot_counter)
    radar.use_legend(loc='lower left', bbox_to_anchor=(0.3, 0, 1, 1), ncol=radar.plot_counter,
                     bbox_transform=matplotlib.transforms.IdentityTransform())


    plt.show()

def get_MAP_simple(red,clusters_names,n=200000): #esta función es exactamente igual que get_MAP pero en este caso no calculamos las probabilidades P(valor MAP variable | C)
                                                #de esta forma solo obtenemos los representantes en un Dataframe
    MAP = pd.DataFrame()
    sample=joint_sampling(red,clusters_names,n)

    variables = pb.Dag(red.nodes(), red.arcs()).topological_sort()
    variables.remove('cluster')
    for k in range(len(clusters_names)):
        sample_c=sample.loc[sample['cluster']==clusters_names[k]]
        sample_c=sample_c.drop('cluster',axis=1)
        dict=sample_c.value_counts().to_dict()
        map=list(max(dict, key=dict.get))
        MAP[clusters_names[k]] = map

    MAP = MAP.set_index(np.array(variables))
    print(MAP)

    plot = MAP.T
    return plot

def df_to_dict(dataframe):  #con esta función obtenemos un diccionario que para cada variable contiene un diccionario con la codificación de las categorias en nº enteros
                            #como por ejemplo {fruta:{manzana:0,pera:1}}. Esta codificacion de las variables es necesaria para la funcion de naming_discrete que nos
                            #devuelve un radar chart con los valores que toma cada uno de los representantes.
    initial_dict={}
    for var in dataframe.columns:
        traductor={}
        categories=dataframe[var].unique().tolist()
        categories.sort()
        for i, category in enumerate(categories):
            traductor[category]=i+1
        initial_dict[var]=traductor
    dictionary=dict(sorted(initial_dict.items(), key=lambda item: len(list(item[1].values())),reverse=True))

    return dictionary



def naming_categories(dataframe_map, importance,df_categories): #Esta funcion cumple el mismo cometido que la funcion naming a diferencia de que en este caso el radar chart
                                                                #mostrado enseña los valores que toma cada representante en vez de P(valor MAP variable | C)
                                                                #es por ello que necesitamos como input las categorias de las variables codificadas (df_categories)

    fig1 = plt.figure(figsize=(10, 10))
    radar = radar_chart_discrete_categories.ComplexRadar(fig1, df_categories, show_scales=True)


    for g in dataframe_map.index:
        info = f"cluster {g}"
        for var in importance[g].keys():
            info = info + f"\n {var, dataframe_map.loc[g][var]} importance {importance[g][var]}"
        radar.plot(dataframe_map.loc[g], df_categories,label=info )
        radar.fill(dataframe_map.loc[g], df_categories,alpha=0.5)

    radar.set_title("MAP representative of each cluster")
    radar.use_legend(loc='lower left', bbox_to_anchor=(0.3, 0, 1, 1), ncol=radar.plot_counter,
                     bbox_transform=matplotlib.transforms.IdentityTransform())


    plt.show()



def posterior_probability(red, clusters_names, df_categories, point):  # Esta función nos devuelve P(C| x). Point debe ir en orden ancestral
    lklh = []
    ancestral_order = pb.Dag(red.nodes(), red.arcs()).topological_sort()
    ancestral_order.remove('cluster')

    for p_c in clusters_names: #para cada cluster obtenemos P(c,x). Como P(c|x)=P(c,x)/P(x) y C es categórica simplemente necesitamos obtener P(c,x) y normalizar
        #para obtener la distribución a posteriori

        instance = pd.DataFrame(columns=ancestral_order)
        i = 0
        for column in instance.columns:
            instance[column] = pd.Series(pd.Categorical([point[i]], categories=df_categories[column]))
            i = i + 1
        instance['cluster'] = pd.Series(pd.Categorical([p_c], categories=clusters_names))
        x = math.exp(red.logl(instance)[0])

        lklh.append(x)
    t = sum(lklh)
    prob_posteriors = [x / t for x in lklh]

    return prob_posteriors

#Distancia de Hellinger para distribuciones discretas
def hellinger_distance(p,q):
    d=0
    for i in range(len(p)):
        d=d+(math.sqrt(p[i])-math.sqrt(q[i]))**2
    return math.sqrt(d)/math.sqrt(2)




def ev_probability(bn, instances,cluster_names,df_categories, n=1000):  #Esta función utiliza likelihood weighting para obtener P(e)
    evidence = list(instances.keys())
    evidence_value = instances.copy()

    w = 0 #a esta variable se el irán sumando los pesos w a medida que se calculen

    ancestral_order = pb.Dag(bn.nodes(), bn.arcs()).topological_sort()

    for i in range(n): #obtenemos la muestra para calcular P(e)
        sample = evidence_value.copy()
        for var in ancestral_order:
            if bn.num_parents(var) == 0: #Con esto fijamos que la variable sea la variable cluster la cual es la única sin padres y sampleamos
                sample[var] = numpy.random.choice(cluster_names, 1,
                                                     p=from_CPT_to_df(str(bn.cpd('cluster'))).iloc[0].tolist())[0]
            else: #sampleamos de aquellas variables que no estén en la evidencia
                if var not in evidence_value.keys():
                    prob = from_CPT_to_df(str(bn.cpd(var)))
                    for element in bn.parents(var):
                        prob = prob.loc[prob[element] == itemgetter(element)(sample)]
                        prob = prob.drop(element, axis=1)
                    cat = prob.columns
                    prob = prob.iloc[0].tolist()
                    prob = [float(x) for x in prob]
                    t = sum(prob)
                    sample[var] = numpy.random.choice(cat, 1,
                                                         p=[x / t for x in prob])[0]


        loglikelihood = 1
        for ev in evidence: #obtenemos el peso w para el sample obtenido que queda definido en la variable loglikelihood y se lo sumamos a w
            parents = pd.DataFrame(columns=list(bn.parents(ev)))
            parents[ev] = pd.Series(pd.Categorical([itemgetter(ev)(sample)], categories=df_categories[ev]))
            for element in bn.parents(ev):
                parents[element] = pd.Series(pd.Categorical([itemgetter(element)(sample)], categories=df_categories[element]))
            x = bn.cpd(ev).logl(parents)

            loglikelihood = loglikelihood * math.exp(x[0])

        w = w + loglikelihood

    if w == 0:

        return 0
    else:
        evidence_probability = w / n
        return evidence_probability



def importance_1(red, point, df_categories, clusters_names):
    """Importancia de la variable mediante la variación de la propia variable
       point en orden ancestral.
    """

    # Ojo: 'cluster' debe estar fuera de variables
    variables = red.nodes()
    variables.remove('cluster')

    # Calcula la probabilidad posterior del MAP
    prob_posterior_map = posterior_probability(red, clusters_names, df_categories, point)
    print("[DEBUG] prob_posterior_map =", prob_posterior_map)

    # Orden ancestral
    ancestral_order = pb.Dag(red.nodes(), red.arcs()).topological_sort()
    ancestral_order.remove('cluster')

    # Diccionario de importancias
    importance = {}

    for k in range(len(ancestral_order)):
        var = ancestral_order[k]

        # Construimos "instances" para ev_probability(...) sin la variable actual
        instances = {}
        lista = red.nodes()
        lista.remove('cluster')
        lista.remove(var)
        for variable in lista:
            # Toma el valor actual del MAP (point) para esa variable
            idx_in_point = ancestral_order.index(variable)
            instances[variable] = point[idx_in_point]

        # Calculamos P(X_(-i)) => e_prob
        e_prob = ev_probability(red, instances, clusters_names, df_categories)
        print(f"[DEBUG] var={var}: e_prob={e_prob:.6f}, df_categories[var]={df_categories[var]}")

        # Recorremos categorías de la variable para ver la diferencia con point[k]
        distances = []
        for category in df_categories[var]:
            # Debug: mostrar la categoría que estamos analizando
            print(f"[DEBUG] var={var}, category={category}, MAP_category={point[k]}")
            if point[k] != category:
                # ========== Cálculo de la Hellinger distance con esa categoría ==========
                evidence = point.copy()
                evidence[k] = category

                # Calculamos prob posterior condicional P(C | esa evidencia)
                lklh = []
                for p_c in clusters_names:
                    instance = pd.DataFrame(columns=ancestral_order)
                    i = 0
                    for column in instance.columns:
                        instance[column] = pd.Series(pd.Categorical([evidence[i]], 
                                                    categories=df_categories[column]))
                        i += 1

                    instance['cluster'] = pd.Series(pd.Categorical([p_c], 
                                                categories=clusters_names))
                    x = math.exp(red.logl(instance)[0])
                    lklh.append(x)

                # normalizar
                t = sum(lklh)
                prob_posterior = [x / t for x in lklh]

                # Distancia de Hellinger respecto a la posterior del MAP
                d = hellinger_distance(prob_posterior_map, prob_posterior)

                # ===== Calculamos P(X_i, X_(-i)) =====
                probability = 0
                for p_c in clusters_names:
                    data = pd.DataFrame()
                    for inst_var, inst_val in instances.items():
                        data[inst_var] = pd.Series(pd.Categorical([inst_val], 
                                                categories=df_categories[inst_var]))
                    data[var] = pd.Series(pd.Categorical([category], 
                                         categories=df_categories[var]))
                    data['cluster'] = pd.Series(pd.Categorical([p_c], 
                                             categories=clusters_names))

                    probability += math.exp(red.logl(data)[0])

                # Añadimos el término a distances
                product_val = d * probability
                distances.append(product_val)

                print(f"[DEBUG] var={var}, category={category}, d={d:.6f}, prob_sum={probability:.6f}, product={product_val:.6f}")
            else:
                print(f"[DEBUG] var={var}, category={category} == MAP_category => skipping.")

        # Al terminar, revisamos si distances quedó vacío
        print(f"[DEBUG] var={var}, final distances={distances}")

        if not distances:
            # Si no hay datos para calcular la media (caso 1 sola categoría o no se cumplió if)
            print(f"[WARNING] var={var} => distances vacío; asignando 0 a importancia.")
            importance[var] = 0
        else:
            importance[var] = mean(distances) / e_prob
            print(f"[DEBUG] var={var}, importance={importance[var]:.6f}")

    return importance

