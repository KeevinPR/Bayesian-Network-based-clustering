import pandas as pd
import pybnesian as pb
from itertools import product
import math
import numpy as np
import random
from multiprocessing import Pool
import itertools
from concurrent.futures import ProcessPoolExecutor

def EM(red, dataframe, clusters_names, kmax=100):
#def EM(red, dataframe, clusters_names, kmax=10):
    print("[DEBUG] Entering EM(...)")
    print(f"[DEBUG] EM parameters: kmax={kmax}, #clusters={len(clusters_names)}, dataframe.shape={dataframe.shape}")
    k = 0
    rb = red.clone()

    df = dataframe.copy()
    # Instanciamos los parámetros de la variable cluster, 
    # para parámetros sampleados de una uniforme.
<<<<<<< HEAD
    print("[DEBUG] EM: Initializing cluster distribution randomly.")
    probability_distribution = [np.random.uniform() for _ in range(len(clusters_names))]
    ini_clust = choice(
        clusters_names, df.shape[0],
        p=[x / sum(probability_distribution) for x in probability_distribution]
    )
=======

    probability_distribution = [np.random.uniform() for i in range(len(clusters_names))]
    ini_clust = weighted_choice(clusters_names, probability_distribution)

>>>>>>> ad00e2d (Updated customers.py and discrete_hellinger and discrete_structure with preprocessing and parallelized sem function)

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

    # Con esta muestra sampleada hacemos el primer fit de la red utilizando pybnesian.
    rb.fit(df)
    print("[DEBUG] EM: Done initial fit with random cluster assignment.")

    # Comenzamos las iteraciones del algoritmo
    while k < kmax:
        if k % 10 == 0:
            print(f"[DEBUG] EM iteration k={k}/{kmax}")

<<<<<<< HEAD
        new_c = []  # nueva muestra sampleada de P(C|X)
        logl = []   # listas de loglikelihoods para cada punto con cada cluster
        df = dataframe.copy()

        # para cada cluster añadimos una columna 'cluster' y calculamos loglikelihood
        for cluster in clusters_names:
            c = [cluster]*df.shape[0]
            df['cluster'] = pd.Series(pd.Categorical(c, categories=clusters_names))
            log_vals = rb.logl(df).tolist()
            logl.append(log_vals)

        # E-step: sampleamos una nueva asignación de clusters
        for row_idx in range(df.shape[0]):
            lklh = []
            for c_idx in range(len(clusters_names)):
                x_val = math.exp(logl[c_idx][row_idx])
                lklh.append(x_val)
            total = sum(lklh)
            prob_posterior = [val / total for val in lklh]
            sampled_cluster = np.random.choice(np.asarray(clusters_names), p=prob_posterior)
            new_c.append(sampled_cluster)

        df['cluster'] = new_c
        # completar dataset
        categor = []
        for var in dataframe.columns:
            categor.append(dataframe[var].cat.categories.tolist())
=======
        df = dataframe.copy()
        logl = parallel_compute_logl(clusters_names, df, rb) # esta lista contiene las listas de los loglikelihoods para cada punto del dataframe con cada posible cluster: P(c'k',X)
        # Este loglikelihood se utilizará para el cálculo de P(C|X)=P(C,X)/P(X). Como C es categórica con elevar e al loglikelhood tenemos P(C,X) y normalizando P(C|X)
        new_c = parallel_sample_clusters(df, logl, clusters_names)  # esta lista va a contener la nueva muestra sampleada de P(C|X) en el paso expectation  

        df['cluster'] = new_c

        #Dataframe completation
        # Prepare categories for combinations
        categor = [dataframe[var].cat.categories.tolist() for var in dataframe.columns]
>>>>>>> ad00e2d (Updated customers.py and discrete_hellinger and discrete_structure with preprocessing and parallelized sem function)
        categor.append(clusters_names)

        # Generate combinations in parallel
        x = parallel_generate_combinations(categor, df.columns, num_chunks=10)

        # Concatenate the generated combinations with the original DataFrame
        df = pd.concat([df, x], ignore_index=True)

        for var in df.columns:
            df[var] = df[var].astype('category')

        # M-step: fit de la red
        rb.fit(df)
        # Optional second fit
        rb_temp = red.clone()
        rb_temp.fit(df)
        rb = rb_temp

        k += 1

    print(f"[DEBUG] Exiting EM(...). Final iteration was k={k}.")
    return rb


<<<<<<< HEAD
def structure_logl(red, dataframe, clusters_names, sample=20):
    print("[DEBUG] Entering structure_logl(...)")
=======
'''def structure_logl(red, dataframe, clusters_names, sample=20): #esta función estima el expected loglikelihood de los datos.
    posterioris_x = []
>>>>>>> ad00e2d (Updated customers.py and discrete_hellinger and discrete_structure with preprocessing and parallelized sem function)
    rb = red.clone()
    df = dataframe.copy()
    logl = []  
    slogl = []  # suma de loglikelihood para cada muestra

    # Para cada cluster, añadimos la columna 'cluster' para calcular P(c'k',X)
    for cluster in clusters_names:
        c = [cluster]*df.shape[0]
        df['cluster'] = pd.Series(pd.Categorical(c, categories=clusters_names))
        logl_vals = rb.logl(df).tolist()
        logl.append(logl_vals)

    # Calculamos los posteriores P(C|X)
    posterioris_x = []
    for row_idx in range(df.shape[0]):
        lklh = []
        for c_idx, _ in enumerate(clusters_names):
            x_val = math.exp(logl[c_idx][row_idx])
            lklh.append(x_val)
        total = sum(lklh)
        prob_posterior = [val / total for val in lklh]
        posterioris_x.append(prob_posterior)

    # Realizamos 'sample' para estimar la loglikelihood esperada
    for s_i in range(sample):
        if s_i % 5 == 0:
            print(f"[DEBUG] structure_logl: sample iteration s_i={s_i}/{sample}")
        new_c = []
        df = dataframe.copy()
        for row_idx in range(df.shape[0]):
            new_c.append(np.random.choice(np.asarray(clusters_names), p=posterioris_x[row_idx]))
        df['cluster'] = pd.Series(pd.Categorical(new_c, categories=clusters_names))

        slogl.append(red.slogl(df))

<<<<<<< HEAD
    avg_slogl = sum(slogl) / len(slogl)
    print("[DEBUG] Leaving structure_logl(...)")
    return avg_slogl
=======
    return sum(slogl) / len(slogl)'''
>>>>>>> ad00e2d (Updated customers.py and discrete_hellinger and discrete_structure with preprocessing and parallelized sem function)


def n_param(red, number_of_clusters, categories_df):
    print("[DEBUG] Entering n_param(...)")
    # Dado una red, el nº de clusters y las categorías de cada variable,
    # calculamos el nº de parámetros estimados.
    n = number_of_clusters - 1  # variable cluster => (clusters-1) parámetros
    for var in red.children('cluster'):
        n += (red.num_parents(var)) * (len(categories_df[var]) - 1)
    print(f"[DEBUG] n_param returning {n}")
    return n

<<<<<<< HEAD
#def sem(bn, dataframe, categories_df, clusters_names, max_iter=2, em_kmax=500, structlog_sample=500):
def sem(bn, dataframe, categories_df, clusters_names, max_iter=2, em_kmax=5, structlog_sample=5):
    print("[DEBUG] Entering sem(...)")
    print(f"[DEBUG] sem parameters: max_iter={max_iter}, em_kmax={em_kmax}, structlog_sample={structlog_sample}")
    print("[DEBUG] Running initial EM (Naive Bayes).")
    # Comenzamos estimando la red naive bayes
    clgbn = EM(bn, dataframe, clusters_names, em_kmax)
=======

#Parallelized with Pool
'''def structure_logl(M_h, M_n_h, dataframe, clusters_names): #esta función estima el expected loglikelihood de los datos.
    """
    Parallelized computation of structure_logl.
    """
    rb_n_h = M_n_h.clone()
    rb_h = M_h.clone()
    df = dataframe.copy()

    # Parallelize the computation for each row
    with Pool() as pool:
        row_sums = pool.starmap(
            compute_row_contribution,
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
    with Pool() as pool:
        row_sums = pool.starmap(
            compute_row_contribution,
            [(row_index, df.copy(), rb_n_h, rb_h, clusters_names) for row_index in range(df.shape[0])]
        )

    # Aggregate the results
    structure_logl = sum(row_sums)
    return structure_logl

def sem(bn, dataframe, categories_df, clusters_names, max_iter=2, em_kmax=500):
    clgbn = EM(bn, dataframe, clusters_names, em_kmax)  # comenzamos estimando la red naive bayes
>>>>>>> ad00e2d (Updated customers.py and discrete_hellinger and discrete_structure with preprocessing and parallelized sem function)
    best = clgbn.clone()
    i = 0
    df = dataframe.copy()
<<<<<<< HEAD
=======
    BIC = -2 * structure_logl(clgbn, clgbn, df, clusters_names) + math.log(df.shape[0]) * n_param(clgbn,len(clusters_names),categories_df)  # bic de la primera red naive
>>>>>>> ad00e2d (Updated customers.py and discrete_hellinger and discrete_structure with preprocessing and parallelized sem function)

    # BIC de la primera red naive
    initial_logl = structure_logl(clgbn, df, clusters_names, structlog_sample)
    BIC = -2*initial_logl + math.log(df.shape[0]) * n_param(clgbn, len(clusters_names), categories_df)
    print(f"[DEBUG] Initial BIC={BIC}")

    participant_nodes = list(df.columns.copy())
    possible_arcs = list(itertools.permutations(participant_nodes, 2))
    print(f"[DEBUG] Number of possible arcs to check: {len(possible_arcs)}")

    # Algoritmo sem
    while i < max_iter:
        print(f"[DEBUG] SEM outer loop iteration i={i}/{max_iter}. Current BIC={BIC}")
        s = 0
        k = 0
        random.shuffle(possible_arcs)

        # 1) Intentamos añadir arcos
        while k < len(possible_arcs):
<<<<<<< HEAD
            p_node, c_node = possible_arcs[k]
            if clgbn.can_add_arc(p_node, c_node):
                print(f"[DEBUG] Trying add_arc({p_node}->{c_node}). k={k}")
=======

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
>>>>>>> ad00e2d (Updated customers.py and discrete_hellinger and discrete_structure with preprocessing and parallelized sem function)
                red = pb.DiscreteBN(nodes=clgbn.nodes(), arcs=clgbn.arcs())
                red.add_arc(p_node, c_node)
                red = EM(red, df, clusters_names, em_kmax)
<<<<<<< HEAD
=======
                l = -2 * structure_logl(red, clgbn, df, clusters_names) + math.log(df.shape[0]) * n_param(red, len(clusters_names), categories_df)
>>>>>>> ad00e2d (Updated customers.py and discrete_hellinger and discrete_structure with preprocessing and parallelized sem function)

                new_logl = structure_logl(red, df, clusters_names, structlog_sample)
                new_BIC = -2*new_logl + math.log(df.shape[0])*n_param(red, len(clusters_names), categories_df)
                if new_BIC >= BIC:
                    k += 1
                else:
                    print(f"[DEBUG] add_arc improved BIC from {BIC} to {new_BIC}")
                    BIC = new_BIC
                    clgbn = red
                    best = clgbn.clone()
                    s += 1
                    k = len(possible_arcs)  # break out
            else:
                k += 1

        # 2) Intentamos invertir arcos
        k = 0
        possible = list(clgbn.arcs())
        for element in participant_nodes:
            if ('cluster', element) in possible:
                possible.remove(('cluster', element))
        random.shuffle(possible)

        while k < len(possible):
<<<<<<< HEAD
            p_node, c_node = possible[k]
            if clgbn.can_flip_arc(p_node, c_node):
                print(f"[DEBUG] Trying flip_arc({p_node}->{c_node}). k={k}")
                red = pb.DiscreteBN(nodes=clgbn.nodes(), arcs=clgbn.arcs())
                red.flip_arc(p_node, c_node)
                red = EM(red, df, clusters_names, em_kmax)
=======
            red = pb.DiscreteBN(nodes=clgbn.nodes(), arcs=clgbn.arcs())
            red.remove_arc(possible[k][0], possible[k][1])
            red = EM(red, df, clusters_names, em_kmax)
            l = -2 * structure_logl(red, clgbn, df, clusters_names) + math.log(df.shape[0]) * n_param(red,len(clusters_names),categories_df)
>>>>>>> ad00e2d (Updated customers.py and discrete_hellinger and discrete_structure with preprocessing and parallelized sem function)

                new_logl = structure_logl(red, df, clusters_names, structlog_sample)
                new_BIC = -2*new_logl + math.log(df.shape[0])*n_param(red, len(clusters_names), categories_df)
                if new_BIC >= BIC:
                    k += 1
                else:
                    print(f"[DEBUG] flip_arc improved BIC from {BIC} to {new_BIC}")
                    BIC = new_BIC
                    clgbn = red
                    best = clgbn.clone()
                    s += 1
                    k = len(possible)
            else:
                k += 1

        # 3) Intentamos eliminar arcos
        k = 0
        possible = list(clgbn.arcs())
        for element in participant_nodes:
            if ('cluster', element) in possible:
                possible.remove(('cluster', element))
        random.shuffle(possible)

        while k < len(possible):
            p_node, c_node = possible[k]
            print(f"[DEBUG] Trying remove_arc({p_node}->{c_node}). k={k}")
            red = pb.DiscreteBN(nodes=clgbn.nodes(), arcs=clgbn.arcs())
            red.remove_arc(p_node, c_node)
            red = EM(red, df, clusters_names, em_kmax)

            new_logl = structure_logl(red, df, clusters_names, structlog_sample)
            new_BIC = -2*new_logl + math.log(df.shape[0])*n_param(red, len(clusters_names), categories_df)
            if new_BIC >= BIC:
                k += 1
            else:
                print(f"[DEBUG] remove_arc improved BIC from {BIC} to {new_BIC}")
                BIC = new_BIC
                clgbn = red
                best = clgbn.clone()
                s += 1
                k = len(possible)

        print(f"[DEBUG] End of iteration i={i}, BIC={BIC}, s={s}")
        if s == 0:
            i += 1
        else:
            i = 0

<<<<<<< HEAD
    print("[DEBUG] Exiting sem(...). Returning best BN.")
    return best
=======
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
    with Pool() as pool:  # Automatically uses all available cores
        # Generate chunks of combinations
        chunks = pool.starmap(generate_combinations_chunk, [(categor, start, end) for start, end in ranges])
        # Process each chunk into a DataFrame
        dataframes = pool.starmap(process_combinations_chunk, [(chunk, df_columns) for chunk in chunks])

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
    with ProcessPoolExecutor() as executor:
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
    with Pool() as pool:
        # Distribute the computation of log-likelihoods across all clusters
        results = pool.starmap(
            compute_logl_for_cluster,
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
    with Pool() as pool:
        # Distribute the computation across all rows
        new_c = pool.starmap(
            compute_posterior_and_sample,
            [(row_index, logl, clusters_names) for row_index in range(df.shape[0])]
        )
    return new_c'''

def parallel_sample_clusters(df, logl, clusters_names):
    """
    Parallelize the computation of P(C|x) and sampling for all rows using ProcessPoolExecutor.
    """
    with ProcessPoolExecutor() as executor:
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
        df['cluster'] = pd.Series([cluster] * df.shape[0], dtype="category", categories=clusters_names)
        unnormalized_probs.append(math.exp(rb_n_h.logl(df.iloc[[row_index]]).iloc[0]))

    # Normalize posterior probabilities
    total_prob = sum(unnormalized_probs)
    posterior_probs = [prob / total_prob for prob in unnormalized_probs]

    # Compute the sum of contributions for the row
    row_sum = 0
    for cluster, posterior_prob in zip(clusters_names, posterior_probs):
        df['cluster'] = pd.Series([cluster] * df.shape[0], dtype="category", categories=clusters_names)
        logl_h = rb_h.logl(df.iloc[[row_index]]).iloc[0]
        row_sum += posterior_prob * logl_h

    return row_sum
>>>>>>> ad00e2d (Updated customers.py and discrete_hellinger and discrete_structure with preprocessing and parallelized sem function)
