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
from multiprocessing import Pool

from operator import itemgetter
import math

# Cache global de tablas CPT para no recalcular from_CPT_to_df cada vez
_CPD_CACHE = {}

def get_cpd_df(bn, var_name):
    """
    Devuelve la CPT de 'var_name' como DataFrame, cacheando el resultado.
    La clave usa id(bn) para evitar mezclar redes distintas.
    """
    key = (id(bn), var_name)
    if key not in _CPD_CACHE:
        _CPD_CACHE[key] = from_CPT_to_df(str(bn.cpd(var_name)))
    return _CPD_CACHE[key]

def get_ancestral_order_without_cluster(red):
    order = pb.Dag(red.nodes(), red.arcs()).topological_sort()
    if 'cluster' in order:
        order.remove('cluster')
    return order



def joint_sampling(red, cluster_names, n=100000):
    """
    Efficiently sample n instances from the joint distribution P(cluster, X)
    using pybnesian's internal sampler. This is conceptually identical to
    the previous implementation but avoids Python loops and multiprocessing
    overhead.
    """
    # red.sample(n) devuelve un objeto que podemos convertir a pandas
    samples = red.sample(n).to_pandas()

    # Aseguramos que 'cluster' es categórica con el orden correcto
    if "cluster" in samples.columns:
        samples["cluster"] = pd.Categorical(samples["cluster"],
                                            categories=cluster_names)

    return samples


#SE puede paralelizar para cada cluster
def get_MAP(red, clusters_names, n=2000):
    """
    Compute MAP representatives for each cluster using Monte Carlo samples
    from P(cluster, X). For each cluster and variable we return a tuple
    (most_probable_category, P(X_j = category | cluster)).
    """
    # Muestra conjunta P(cluster, X)
    sample = joint_sampling(red, clusters_names, n)

    # Orden ancestral y eliminación de 'cluster'
    variables = pb.Dag(red.nodes(), red.arcs()).topological_sort()
    variables.remove("cluster")

    MAP = pd.DataFrame(index=variables)

    for c in clusters_names:
        # Nos quedamos con las filas de ese cluster
        sample_c = sample.loc[sample["cluster"] == c, variables]

        if sample_c.empty:
            # Si por algún motivo no hay muestras para ese cluster,
            # evitamos un error y ponemos NaN.
            MAP[c] = [(np.nan, 0.0)] * len(variables)
            continue

        # Configuración conjunta más frecuente (MAP conjunto)
        mode_tuple = sample_c.value_counts().idxmax()
        mode_values = list(mode_tuple)

        # Para cada variable guardamos (valor_MAP, probabilidad_condicional)
        map_with_probs = []
        for var, v in zip(variables, mode_values):
            p_v = (sample_c[var] == v).mean()
            map_with_probs.append((v, float(p_v)))

        MAP[c] = map_with_probs

    return MAP.T  # filas = clusters, columnas = variables




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



def posterior_probability(red, clusters_names, df_categories, point):
    """
    Compute P(C | x) for a given point x.
    `point` must be in ancestral order (excluding 'cluster').
    """
    """
    Compute P(C | x) for a given point x.
    `point` must be in ancestral order (excluding 'cluster').
    """
    ancestral_order = get_ancestral_order_without_cluster(red)

    # Construimos un DataFrame con una fila por valor de cluster
    data = pd.DataFrame([point] * len(clusters_names), columns=ancestral_order)

    # Aseguramos categorías correctas por columna
    for col in ancestral_order:
        cats = df_categories[col]
        # df_categories[col] puede ser dict {categoria: indice} o lista de categorías
        if isinstance(cats, dict):
            cats = list(cats.keys())
        data[col] = pd.Categorical(data[col], categories=cats)

    data["cluster"] = pd.Categorical(clusters_names, categories=clusters_names)

    # logl acepta múltiples filas → vectorizamos
    logl = red.logl(data)
    lklh = np.exp(np.array(logl))

    t = lklh.sum()
    if t == 0:
        # Evitar división por cero en casos numéricamente degenerados
        return [1.0 / len(clusters_names)] * len(clusters_names)

    return (lklh / t).tolist()

#Distancia de Hellinger para distribuciones discretas
def hellinger_distance(p, q):
    """
    Compute the Hellinger distance between two discrete distributions using NumPy.
    """
    p = np.sqrt(p)
    q = np.sqrt(q)
    return np.sqrt(np.sum((p - q) ** 2)) / np.sqrt(2)




def ev_probability(bn, instances,cluster_names,df_categories, n=80):  #Esta función utiliza likelihood weighting para obtener P(e)
    evidence = list(instances.keys())
    evidence_value = instances.copy()

    w = 0 #a esta variable se el irán sumando los pesos w a medida que se calculen

    ancestral_order = get_ancestral_order_without_cluster(bn)

    for i in range(n): #obtenemos la muestra para calcular P(e)
        sample = evidence_value.copy()
        # Explicitly sample 'cluster' if it's missing (since we removed it from ancestral_order)
        if 'cluster' not in sample:
             cluster_cpt = get_cpd_df(bn, 'cluster')
             sample['cluster'] = numpy.random.choice(
                cluster_names,
                1,
                p=cluster_cpt.iloc[0].tolist()
            )[0]

        for var in ancestral_order:
            if bn.num_parents(var) == 0:
                 continue # Already handled cluster or other roots
            else: #sampleamos de aquellas variables que no estén en la evidencia
                if var not in evidence_value.keys():
                    prob = get_cpd_df(bn, var)
                    for element in bn.parents(var):
                        prob = prob.loc[prob[element] == itemgetter(element)(sample)]
                        prob = prob.drop(element, axis=1)
                    cat = prob.columns
                    prob = prob.iloc[0].tolist()
                    prob = [float(x) for x in prob]
                    sample[var] = weighted_choice(cat, prob)  # sampleamos la variable


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


def importance_1(red,point,df_categories,clusters_names):  #importancia de la variable a través de variación de la propia variable
                                                           #point en orden ancestral

    variables = red.nodes()
    variables.remove('cluster')
    prob_posterior_map = posterior_probability(red, clusters_names, df_categories, point)#calculamos P(C | MAP)

    ancestral_order = get_ancestral_order_without_cluster(red)

    importance = {}
    for k in range(len(ancestral_order)): #calculamos la importancia para cada variable.
        distances = []
        var = ancestral_order[k]
        instances = {}
        lista = red.nodes()
        lista.remove('cluster')
        lista.remove(var)
        for variable in lista:
            instances[variable] = point[ancestral_order.index(variable)]

        e_prob = ev_probability(red, instances, clusters_names,df_categories) #Como P(X_i | X_(-i))=P(X_i,X_(-i))/P(X_(-i)) calculamos P(X_(-i))
        # Use parallel_compute_distances to compute the mean of distances
        mean_distance = parallel_compute_distances(
            red, point, df_categories, clusters_names, prob_posterior_map, instances, var
        )

        # Compute the importance for the variable
        importance[var] = mean_distance / e_prob

        
    return importance



def weighted_choice(elements, weights):
    cumulative_weights = np.cumsum(weights)
    total = cumulative_weights[-1]
    random_value = np.random.uniform(0, total)
    return elements[np.searchsorted(cumulative_weights, random_value)]















def compute_distance_for_category(category, point, k, red, clusters_names, df_categories, prob_posterior_map, instances, var):
    """
    Compute the distance for a single category.
    """
    if point[k] == category:
        return 0  # Skip if the category is the same as the current value in the point

    # Update the evidence with the new category
    evidence = point.copy()
    evidence[k] = category

    # Compute P(C|X) for the updated evidence
    lklh = []
    # Compute P(C|X) for the updated evidence
    lklh = []
    # ancestral_order is not needed here for DataFrame columns if we assume consistent order or pass it
    # But to be safe and consistent with previous code:
    ancestral_order = get_ancestral_order_without_cluster(red)

    for p_c in clusters_names:
        instance = pd.DataFrame(columns=ancestral_order)
        for i, column in enumerate(instance.columns):
            instance[column] = pd.Series(pd.Categorical([evidence[i]], categories=df_categories[column]))
        instance['cluster'] = pd.Series(pd.Categorical([p_c], categories=clusters_names))
        x = math.exp(red.logl(instance)[0])
        lklh.append(x)

    t = sum(lklh)
    prob_posterior = [x / t for x in lklh]

    # Compute the Hellinger distance
    d = hellinger_distance(prob_posterior_map, prob_posterior)

    # Compute P(X_i, X_(-i)) and multiply by the distance
    probability = 0
    for p_c in clusters_names:
        data = pd.DataFrame()
        for instance_key in instances.keys():
            data[instance_key] = pd.Series(pd.Categorical([instances[instance_key]], categories=df_categories[instance_key]))
        data[var] = pd.Series(pd.Categorical([category], categories=df_categories[var]))
        data['cluster'] = pd.Series(pd.Categorical([p_c], categories=clusters_names))
        probability += math.exp(red.logl(data)[0])

    return d * probability

def parallel_compute_distances(red, point, df_categories, clusters_names, prob_posterior_map, instances, var):
    """
    Sequential computation of distances for all categories of a variable.
    """
    """
    Sequential computation of distances for all categories of a variable.
    """
    ancestral_order = get_ancestral_order_without_cluster(red)
    k = ancestral_order.index(var)
    categories = df_categories[var]
    distances = []
    for category in categories:
        distances.append(compute_distance_for_category(
            category, point, k, red, clusters_names, df_categories, prob_posterior_map, instances, var
        ))
    
    if not distances:
        return 0.0

    return sum(distances) / len(distances)