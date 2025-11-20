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
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from operator import itemgetter
import math



def joint_sampling(red,cluster_names,n=100000):  #esta funcion realiza un sampling de la Red Bayesiana de forma que obtenemos una muestra de la distribución conjunta de tamaño n
    """
    Parallelized version of joint_sampling using dynamic task scheduling.
    """
    ancestral_order = pb.Dag(red.nodes(), red.arcs()).topological_sort()

    # Use ProcessPoolExecutor to dynamically assign tasks to available cores
    with ProcessPoolExecutor() as executor:
        # Submit tasks for each sample
        futures = [executor.submit(sample_single_instance, i, ancestral_order, red, cluster_names) for i in range(n)]

        # Collect results as they complete
        results = [future.result() for future in futures]

    # Convert the results into a DataFrame
    samples = pd.DataFrame(results)
    return samples


#SE puede paralelizar para cada cluster
def get_MAP(red,clusters_names,n=200000): #Esta función nos devuelve el map dado cada cluster, es decir, el representante de cada cluster. La inferencia se realiza
    """
    Parallelized version of get_MAP.
    """
    MAP = pd.DataFrame()
    sample = joint_sampling(red, clusters_names, n)  # Obtain a sample of the joint distribution

    variables = pb.Dag(red.nodes(), red.arcs()).topological_sort()
    variables.remove('cluster')

    # Parallelize the computation for each cluster
    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(compute_map_for_cluster, k, sample, variables, clusters_names)
                   for k in range(len(clusters_names))]

        # Collect results as they complete
        results = [future.result() for future in futures]

    # Aggregate results into the MAP DataFrame
    for cluster_name, map_values in results:
        MAP[cluster_name] = map_values

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
    """
    Compute P(C|x) for a given point in the Bayesian Network, parallelizing instance generation.
    """
    lklh = []
    ancestral_order = pb.Dag(red.nodes(), red.arcs()).topological_sort()
    ancestral_order.remove('cluster')

    for p_c in clusters_names:  # For each cluster, compute P(c,x)
        # Create the instance DataFrame in parallel
        instance = pd.DataFrame(columns=ancestral_order)
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(generate_instance_column, column, point[i], df_categories[column]): column
                for i, column in enumerate(instance.columns)
            }
            for future in futures:
                column = futures[future]
                instance[column] = future.result()

        # Add the cluster column
        instance['cluster'] = pd.Series(pd.Categorical([p_c], categories=clusters_names))

        # Compute the log-likelihood
        x = math.exp(red.logl(instance)[0])
        lklh.append(x)
    # Normalize to compute posterior probabilities
    t = sum(lklh)
    prob_posteriors = [x / t for x in lklh]

    return prob_posteriors

#Distancia de Hellinger para distribuciones discretas
def hellinger_distance(p, q):
    """
    Compute the Hellinger distance between two discrete distributions using NumPy.
    """
    p = np.sqrt(p)
    q = np.sqrt(q)
    return np.sqrt(np.sum((p - q) ** 2)) / np.sqrt(2)




'''def ev_probability(bn, instances,cluster_names,df_categories, n=1000):  #Esta función utiliza likelihood weighting para obtener P(e)
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
'''

def ev_probability(bn, instances, cluster_names, df_categories, n=1000):
    """
    Compute P(e) using likelihood weighting, parallelizing the sampling process.
    """
    evidence = list(instances.keys())
    evidence_value = instances.copy()

    ancestral_order = pb.Dag(bn.nodes(), bn.arcs()).topological_sort()

    # Parallelize the computation of log-likelihoods
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                compute_loglikelihood_for_sample,
                bn, evidence_value, cluster_names, df_categories, ancestral_order, evidence
            )
            for _ in range(n)
        ]

        # Collect log-likelihoods from all processes
        loglikelihoods = [future.result() for future in futures]

    # Compute the final evidence probability
    w = sum(loglikelihoods)
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

    ancestral_order = pb.Dag(red.nodes(), red.arcs()).topological_sort()
    ancestral_order.remove('cluster')

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



def sample_single_instance(ancestral_order, red, cluster_names):
    """
    Sample a single instance from the Bayesian Network.
    """
    evidence = {}
    for node in ancestral_order:
        if node == 'cluster':  # Sample the cluster variable
            evidence[node] = numpy.random.choice(
                cluster_names, 1,
                p=from_CPT_to_df(str(red.cpd('cluster'))).iloc[0].tolist()
            )[0]
        else:  # Sample other variables based on their parents
            prob = from_CPT_to_df(str(red.cpd(node)))
            for element in red.parents(node):  # Reduce the DataFrame to a single row
                prob = prob.loc[prob[element] == itemgetter(element)(evidence)]
                prob = prob.drop(element, axis=1)
            cat = prob.columns  # Possible categories for the variable
            prob = prob.iloc[0].tolist()
            prob = [float(x) for x in prob]
            evidence[node] = weighted_choice(cat, prob)  # Sample the variable

    return evidence


def compute_map_for_cluster(k, sample, variables, clusters_names):
    """
    Compute the MAP for a single cluster.
    """
    # Filter samples for the current cluster
    sample_c = sample.loc[sample['cluster'] == clusters_names[k]]
    sample_c = sample_c.drop('cluster', axis=1)

    # Compute the MAP
    dict_counts = sample_c.value_counts().to_dict()
    map_values = list(max(dict_counts, key=dict_counts.get))

    # Compute probabilities for each variable
    for i in range(len(variables)):
        var = variables[i]
        p = sample_c.loc[sample_c[var] == map_values[i]][var].tolist().count(map_values[i]) / sample_c.shape[0]
        map_values[i] = (map_values[i], p)

    return clusters_names[k], map_values


def generate_instance_column(column, value, categories):
    """
    Generate a single column for the instance DataFrame.
    """
    return pd.Series(pd.Categorical([value], categories=categories))


def compute_loglikelihood_for_sample(bn, evidence_value, cluster_names, df_categories, ancestral_order, evidence):
    """
    Generate a sample and compute its log-likelihood weight.
    """
    # Copy the evidence values
    sample = evidence_value.copy()

    # Generate the sample
    for var in ancestral_order:
        if bn.num_parents(var) == 0:  # Sample the cluster variable
            sample[var] = np.random.choice(
                cluster_names, 1,
                p=from_CPT_to_df(str(bn.cpd('cluster'))).iloc[0].tolist()
            )[0]
        else:  # Sample variables not in the evidence
            if var not in evidence_value.keys():
                prob = from_CPT_to_df(str(bn.cpd(var)))
                for element in bn.parents(var):
                    prob = prob.loc[prob[element] == itemgetter(element)(sample)]
                    prob = prob.drop(element, axis=1)
                cat = prob.columns
                prob = prob.iloc[0].tolist()
                prob = [float(x) for x in prob]
                sample[var] = weighted_choice(cat, prob)  # Sample the variable

    # Compute the log-likelihood weight
    loglikelihood = 1
    for ev in evidence:
        parents = pd.DataFrame(columns=list(bn.parents(ev)))
        parents[ev] = pd.Series(pd.Categorical([itemgetter(ev)(sample)], categories=df_categories[ev]))
        for element in bn.parents(ev):
            parents[element] = pd.Series(pd.Categorical([itemgetter(element)(sample)], categories=df_categories[element]))
        x = bn.cpd(ev).logl(parents)
        loglikelihood *= math.exp(x[0])

    return loglikelihood


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
    ancestral_order = pb.Dag(red.nodes(), red.arcs()).topological_sort()
    ancestral_order.remove('cluster')

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
    Parallelized computation of distances for all categories of a variable.
    """
    k = list(df_categories.keys()).index(var)  # Get the index of the variable in the point
    categories = df_categories[var]

    # Parallelize the computation for each category
    with ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(
                compute_distance_for_category,
                category, point, k, red, clusters_names, df_categories, prob_posterior_map, instances, var
            )
            for category in categories
        ]

        # Collect results as they complete
        distances = [future.result() for future in futures]

    # Compute the mean of distances
    return sum(distances) / len(distances)