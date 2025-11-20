import pandas as pd
import pybnesian as pb
from itertools import product
import math
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor
import itertools

# Helper function to emulate starmap for ThreadPoolExecutor
def thread_starmap(executor, func, args_list):
    futures = [executor.submit(func, *args) for args in args_list]
    return [future.result() for future in futures]

#Tiene completado de dataset
def EM(red, dataframe, clusters_names, kmax=100):
    k = 0
    rb = red.clone()

    df = dataframe.copy()
    # Instanciamos los parámetros de la variable cluster, como se va a hacer stochastic EM en vez de fijar los parámetros simplemente sampleamos una muestra de la variable cluster
    # para parámetros sampleados de una uniforme.

    probability_distribution = [np.random.uniform() for i in range(len(clusters_names))]
    ini_clust = weighted_choice(clusters_names, probability_distribution)


    df['cluster'] = ini_clust
    #Dataframe completation
    # Prepare categories for combinations
    categor = [dataframe[var].cat.categories.tolist() for var in dataframe.columns]
    categor.append(clusters_names)

    # Generate combinations in parallel
    x = parallel_generate_combinations(categor, df.columns, num_chunks=10)

    # Concatenate the generated combinations with the original DataFrame
    df = pd.concat([df, x], ignore_index=True)

    for var in df.columns:
        df[var] = df[var].astype('category')

    # Fit
    # Con esta muestra sampleada hacemos el primer fit de la red utilizando la libreria de pybnesian.
    rb.fit(df)

    # Comenzamos las iteraciones del algoritmo
    while k < kmax:

        df = dataframe.copy()
        logl = parallel_compute_logl(clusters_names, df, rb) # esta lista contiene las listas de los loglikelihoods para cada punto del dataframe con cada posible cluster: P(c'k',X)
        # Este loglikelihood se utilizará para el cálculo de P(C|X)=P(C,X)/P(X). Como C es categórica con elevar e al loglikelhood tenemos P(C,X) y normalizando P(C|X)
        new_c = parallel_sample_clusters(df, logl, clusters_names)  # esta lista va a contener la nueva muestra sampleada de P(C|X) en el paso expectation  

        df['cluster'] = new_c

        #Dataframe completation
        # Prepare categories for combinations
        categor = [dataframe[var].cat.categories.tolist() for var in dataframe.columns]
        categor.append(clusters_names)

        # Generate combinations in parallel
        x = parallel_generate_combinations(categor, df.columns, num_chunks=10)

        # Concatenate the generated combinations with the original DataFrame
        df = pd.concat([df, x], ignore_index=True)

        for var in df.columns:
            df[var] = df[var].astype('category')
        # Fit
        # Con esta muestra sampleada hacemos el fit de la red utilizando la libreria de pybnesian.
        rb.fit(df)
        rb = red.clone()
        rb.fit(df)
        k = k + 1

    return rb


'''def structure_logl(red, dataframe, clusters_names, sample=20): #esta función estima el expected loglikelihood de los datos.
    posterioris_x = []
    rb = red.clone()
    logl = []  # esta lista contiene las listas de los loglikelihoods para cada punto del dataframe con cada posible cluster: P(c'k',X)
    slogl = []  # esta lista contiene la suma de los loglikelihood de cada punto del dataframe para cada muestra de 'clusters' sampleada de P(C|X)
    df = dataframe.copy()

    for cluster in clusters_names:  # para cada cluster añadimos una columna al dataframe con el cluster dado para poder calcular P(c'k',X)

        c = [cluster for i in range(df.shape[0])]
        df['cluster'] = pd.Series(pd.Categorical(c, categories=clusters_names))
        logl.append(rb.logl(df).tolist())

    for row in range(df.shape[0]):
        lklh = []
        for cluster in range(len(clusters_names)):
            x = math.exp(logl[cluster][row])
            lklh.append(x)
        t = sum(lklh)
        prob_posterior = [x / t for x in lklh]
        posterioris_x.append(prob_posterior)

    for i in range(0, sample):
        new_c = []
        df = dataframe.copy()
        for k in range(df.shape[0]):
            new_c.append(np.random.choice(np.asarray(clusters_names), p=posterioris_x[k]))
        new_c = pd.Series(pd.Categorical(new_c, categories=clusters_names))
        df['cluster'] = new_c

        slogl.append(red.slogl(df))

    return sum(slogl) / len(slogl)'''


def n_param(red, number_of_clusters,categories_df):  # dado una red, el nº de clusters y las categorías de cada variable (es inaccesible por pybnesian) calculamos el nº de parámetros estimados
    n = number_of_clusters - 1  # comenzamos con el nº de parámetros de la variable cluster que como bien sabemos es number_clusters-1 debido a que trabajamos con un TAN
    for var in red.children('cluster'):  # como todos son hijos de cluster accedemos a las variables de esta forma.
        n = n + (red.num_parents(var)) * (len(categories_df[var]) - 1)

    return n


#Parallelized with Pool
'''def structure_logl(M_h, M_n_h, dataframe, clusters_names): #esta función estima el expected loglikelihood de los datos.
    """
    Parallelized computation of structure_logl.
    """
    rb_n_h = M_n_h.clone()
    rb_h = M_h.clone()
    df = dataframe.copy()

    # Parallelize the computation for each row
    with ThreadPoolExecutor() as pool:
        row_sums = thread_starmap(pool, compute_row_contribution,
            [(row_index, df.copy(), rb_n_h, rb_h, clusters_names) for row_index in range(df.shape[0])]
        )

    # Aggregate the results
    structure_logl = sum(row_sums)
    return structure_logl'''

def structure_logl(M_h, M_n_h, dataframe, clusters_names): #esta función estima el expected loglikelihood de los datos.
    """
    Parallelized computation of structure_logl.
    """
    rb_n_h = M_n_h.clone()
    rb_h = M_h.clone()
    df = dataframe.copy()

    # Parallelize the computation for each row
    with ThreadPoolExecutor() as pool:
        row_sums = thread_starmap(pool, compute_row_contribution,
            [(row_index, df.copy(), rb_n_h, rb_h, clusters_names) for row_index in range(df.shape[0])]
        )

    # Aggregate the results
    structure_logl = sum(row_sums)
    return structure_logl

def sem(bn, dataframe, categories_df, clusters_names, max_iter=2, em_kmax=50):
    clgbn = EM(bn, dataframe, clusters_names, em_kmax)  # comenzamos estimando la red naive bayes
    best = clgbn.clone()
    i = 0  # controla cuando se llega a un punto estacionario el máximo nº de iteraciones a realizar
    df = dataframe.copy()
    BIC = -2 * structure_logl(clgbn, clgbn, df, clusters_names) + math.log(df.shape[0]) * n_param(clgbn,len(clusters_names),categories_df)  # bic de la primera red naive


    print(BIC)
    participant_nodes = list(df.columns.copy())  # Nodos de la red
    possible_arcs = list(itertools.permutations(participant_nodes,2))  # Posibles parejas de arcos entre los nodos de la red salvo cluster

    # Comenzamos el algoritmo sem
    while i < max_iter:
        s = 0  # controla si se ha mejorado en la iteración
        k = 0  # lo utilizamos para ir recorriendo la lista de posibles acciones (en este caso añadir arco)
        random.shuffle(possible_arcs)  # mezclamos aleatoriamente los posibles arcos que se pueden introducir
        while k < len(possible_arcs):

            if clgbn.can_add_arc(possible_arcs[k][0], possible_arcs[k][1]):  # comprobamos si el arco puede ser añadido
                red = pb.DiscreteBN(nodes=clgbn.nodes(),arcs=clgbn.arcs())  # en caso de que pueda ser añadido generamos una nueva red con el arco añadido, estimamos parámetros y comparamos BIC
                red.add_arc(possible_arcs[k][0], possible_arcs[k][1])
                red = EM(red, df, clusters_names, em_kmax)
                l = -2 * structure_logl(red, clgbn, df, clusters_names) + math.log(df.shape[0]) * n_param(red, len(clusters_names), categories_df)

                if l >= BIC:  # si no mejoramos pasamos al siguiente arco
                    k = k + 1
                else:  # si mejoramos BIC se actualiza y red se actualiza
                    k = len(possible_arcs)
                    BIC = l
                    clgbn = red
                    best = clgbn.clone()
                    s = s + 1
            else:
                k = k + 1  # si no se puede introducir el arco pasamos al siguiente

        k = 0
        possible = list(
            clgbn.arcs())  # En este caso necesitamos trabajar con la lista de arcos existentes ya que tratamos de invertir alguno

        for element in participant_nodes:  # eliminamos los arcos (cluster,variable) ya que estos no deben tocarse
            if ('cluster', element) in possible:
                possible.remove(('cluster', element))
        random.shuffle(possible)  # hacemos un random order de los posibles candidatos

        while k < len(possible):
            if clgbn.can_flip_arc(possible[k][0], possible[k][1]):  # si se puede invertir el arco de nuevo probamos y comparamos si se mejora el BIC
                red = pb.DiscreteBN(nodes=clgbn.nodes(), arcs=clgbn.arcs())
                red.flip_arc(possible[k][0], possible[k][1])
                red = EM(red, df, clusters_names, em_kmax)
                l = -2 * structure_logl(red, clgbn, df, clusters_names) + math.log(df.shape[0]) * n_param(red, len(clusters_names), categories_df)


                if l >= BIC:
                    k = k + 1
                else:
                    k = len(possible)
                    BIC = l
                    clgbn = red
                    best = clgbn.clone()
                    s = s + 1
            else:
                k = k + 1

        k = 0
        possible = list(
            clgbn.arcs())  # de nuevo queremos eliminar arcos por ello los candidatos son los arcos ya existentes salvo los que no se deben eliminar (cluster,variable)
        for element in participant_nodes:
            if ('cluster', element) in possible:
                possible.remove(('cluster', element))
        random.shuffle(possible)

        while k < len(possible):
            red = pb.DiscreteBN(nodes=clgbn.nodes(), arcs=clgbn.arcs())
            red.remove_arc(possible[k][0], possible[k][1])
            red = EM(red, df, clusters_names, em_kmax)
            l = -2 * structure_logl(red, clgbn, df, clusters_names) + math.log(df.shape[0]) * n_param(red,len(clusters_names),categories_df)

            if l >= BIC:
                k = k + 1
            else:
                k = len(possible)
                BIC = l
                clgbn = red
                best = clgbn.clone()
                s = s + 1

        print(BIC)
        if s == 0:  # si no se mejora comienza el contador de maxiter
            i = i + 1
        else:
            i = 0

    return best



def weighted_choice(elements, weights):
    cumulative_weights = np.cumsum(weights)
    total = cumulative_weights[-1]
    random_value = np.random.uniform(0, total)
    return elements[np.searchsorted(cumulative_weights, random_value)]


'''def parallel_generate_combinations(categor, df_columns, num_chunks):
    """
    Generate combinations in parallel by splitting the workload into a specified number of chunks.
    """
    # Determine the total number of combinations
    total_combinations = np.prod([len(cat) for cat in categor])

    # Calculate the size of each chunk
    chunk_size = total_combinations // num_chunks
    ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_chunks)]
    # Ensure the last chunk includes any remaining combinations
    ranges[-1] = (ranges[-1][0], total_combinations)

    # Generate and process chunks in parallel
    with ThreadPoolExecutor() as pool:  # Automatically uses all available cores
        # Generate chunks of combinations
        chunks = thread_starmap(pool, generate_combinations_chunk, [(categor, start, end) for start, end in ranges])
        # Process each chunk into a DataFrame
        dataframes = thread_starmap(pool, process_combinations_chunk, [(chunk, df_columns) for chunk in chunks])

    # Concatenate all DataFrames into a single DataFrame
    return pd.concat(dataframes, ignore_index=True)'''




def parallel_generate_combinations(categor, df_columns, num_chunks):
    """
    Generate combinations in parallel by splitting the workload into a specified number of chunks.
    """
    # Determine the total number of combinations
    total_combinations = np.prod([len(cat) for cat in categor])

    # Calculate the size of each chunk
    chunk_size = total_combinations // num_chunks
    ranges = [(i * chunk_size, (i + 1) * chunk_size) for i in range(num_chunks)]
    # Ensure the last chunk includes any remaining combinations
    ranges[-1] = (ranges[-1][0], total_combinations)

    # Generate and process chunks in parallel
    with ThreadPoolExecutor() as executor:
        # Generate chunks of combinations
        chunks = list(executor.map(
            generate_combinations_chunk,
            [categor] * len(ranges),
            [start for start, _ in ranges],
            [end for _, end in ranges]
        ))

        # Process each chunk into a DataFrame
        dataframes = list(executor.map(
            process_combinations_chunk,
            chunks,
            [df_columns] * len(chunks)
        ))

    # Concatenate all DataFrames into a single DataFrame
    return pd.concat(dataframes, ignore_index=True)


def generate_combinations_chunk(categor, start, end):
    """
    Generate a chunk of combinations from the Cartesian product.
    """
    combinations = product(*categor)
    chunk = [comb for i, comb in enumerate(combinations) if start <= i < end]
    return chunk

def process_combinations_chunk(chunk, df_columns):
    """
    Convert a chunk of combinations into a DataFrame.
    """
    return pd.DataFrame(chunk, columns=df_columns)


def compute_logl_for_cluster(cluster, df, rb, clusters_names):
    """
    Compute the log-likelihood for a single cluster.
    """
    # Create a column with the cluster value for all rows
    c = [cluster] * df.shape[0]
    df['cluster'] = pd.Series(pd.Categorical(c, categories=clusters_names))
    # Compute the log-likelihood for the cluster
    return rb.logl(df).tolist()

def parallel_compute_logl(clusters_names, df, rb):
    """
    Parallelize the computation of log-likelihoods for all clusters.
    """
    with ThreadPoolExecutor() as pool:
        # Distribute the computation of log-likelihoods across all clusters
        results = thread_starmap(pool, compute_logl_for_cluster,
            [(cluster, df.copy(), rb, clusters_names) for cluster in clusters_names]
        )
    return results

def compute_posterior_and_sample(row_index, logl, clusters_names):
    """
    Compute P(C|x) and sample a cluster for a single row.
    """
    lklh = []  # List of likelihoods for each cluster
    for n in range(len(clusters_names)):
        x = math.exp(logl[n][row_index])  # Compute P(C,X) for the cluster
        lklh.append(x)
    # Sample a cluster based on the posterior probabilities
    return weighted_choice(clusters_names, lklh)

'''def parallel_sample_clusters(df, logl, clusters_names):
    """
    Parallelize the computation of P(C|x) and sampling for all rows.
    """
    with ThreadPoolExecutor() as pool:
        # Distribute the computation across all rows
        new_c = thread_starmap(pool, compute_posterior_and_sample,
            [(row_index, logl, clusters_names) for row_index in range(df.shape[0])]
        )
    return new_c'''

def parallel_sample_clusters(df, logl, clusters_names):
    """
    Parallelize the computation of P(C|x) and sampling for all rows using ThreadPoolExecutor.
    """
    with ThreadPoolExecutor() as executor:
        # Submit tasks for each row
        futures = [executor.submit(compute_posterior_and_sample, row_index, logl, clusters_names) for row_index in range(df.shape[0])]

        # Collect results as they complete
        new_c = [future.result() for future in futures]

    return new_c


def compute_row_contribution(row_index, df, rb_n_h, rb_h, clusters_names):
    """
    Compute the sum of contributions for a single row:
    - Compute posterior probabilities for all clusters.
    - Multiply each posterior probability by rb_h.logl(row) and sum the results.
    """
    # Compute unnormalized posterior probabilities
    unnormalized_probs = []
    for cluster in clusters_names:
        df['cluster'] = pd.Series(pd.Categorical([cluster] * df.shape[0], categories=clusters_names))
        unnormalized_probs.append(math.exp(rb_n_h.logl(df.iloc[[row_index]])[0]))

    # Normalize posterior probabilities
    total_prob = sum(unnormalized_probs)
    posterior_probs = [prob / total_prob for prob in unnormalized_probs]

    # Compute the sum of contributions for the row
    row_sum = 0
    for cluster, posterior_prob in zip(clusters_names, posterior_probs):
        df['cluster'] = pd.Series(pd.Categorical([cluster] * df.shape[0], categories=clusters_names))
        logl_h = rb_h.logl(df.iloc[[row_index]])[0]
        row_sum += posterior_prob * logl_h

    return row_sum