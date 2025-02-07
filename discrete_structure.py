import pandas as pd
import pybnesian as pb
import itertools
import math
import numpy as np
from numpy.random import choice
import random

def EM(red, dataframe, clusters_names, kmax=100):
    print("[DEBUG] Entering EM(...)")
    print(f"[DEBUG] EM parameters: kmax={kmax}, #clusters={len(clusters_names)}, dataframe.shape={dataframe.shape}")
    k = 0
    rb = red.clone()

    df = dataframe.copy()
    
    print("[DEBUG] EM: Initializing cluster distribution randomly.")
    probability_distribution = [np.random.uniform() for _ in range(len(clusters_names))]
    ini_clust = choice(clusters_names, df.shape[0],
                       p=[x / sum(probability_distribution) for x in probability_distribution])

    df['cluster'] = ini_clust
    # "Complete" dataset
    categor = []
    for var in dataframe.columns:
        categor.append(dataframe[var].cat.categories.tolist())
    categor.append(clusters_names)
    combinations = list(itertools.product(*categor))
    x = pd.DataFrame(combinations, columns=df.columns)
    df = pd.concat([df, x])
    for var in df.columns:
        df[var] = df[var].astype('category')

    # First fit
    rb.fit(df)
    print("[DEBUG] EM: Done initial fit with random cluster assignment.")

    while k < kmax:
        if k % 10 == 0:
            print(f"[DEBUG] EM iteration k={k}/{kmax}")

        new_c = []
        logl = []
        # Re-clone the original data
        df = dataframe.copy()

        # Compute log-likelihood for each cluster
        for cluster in clusters_names:
            c = [cluster]*df.shape[0]
            df['cluster'] = pd.Series(pd.Categorical(c, categories=clusters_names))

            log_vals = rb.logl(df).tolist()
            logl.append(log_vals)

        # E-step: sample new cluster assignment from posterior
        for row_idx in range(df.shape[0]):
            lklh = []
            for c_idx, _ in enumerate(clusters_names):
                x_val = math.exp(logl[c_idx][row_idx])
                lklh.append(x_val)
            total = sum(lklh)
            # Normalized posterior
            prob_posterior = [val / total for val in lklh]
            sampled_cluster = np.random.choice(np.asarray(clusters_names), p=prob_posterior)
            new_c.append(sampled_cluster)

        df['cluster'] = new_c

        # "Complete" dataset again
        categor = []
        for var in dataframe.columns:
            categor.append(dataframe[var].cat.categories.tolist())
        categor.append(clusters_names)
        combinations = list(itertools.product(*categor))
        x = pd.DataFrame(combinations, columns=df.columns)
        df = pd.concat([df, x])
        for var in df.columns:
            df[var] = df[var].astype('category')

        # M-step: fit again
        rb.fit(df)

        # (Optional) clone and fit again?
        rb_temp = red.clone()
        rb_temp.fit(df)
        rb = rb_temp

        k += 1

    print("[DEBUG] Exiting EM(...). Final iteration was k={k}.")
    return rb


def structure_logl(red, dataframe, clusters_names, sample=20):
    print("[DEBUG] Entering structure_logl(...)")
    rb = red.clone()
    df = dataframe.copy()
    logl = []
    slogl = []

    # log-likelihood for each cluster
    for cluster in clusters_names:
        c = [cluster]*df.shape[0]
        df['cluster'] = pd.Series(pd.Categorical(c, categories=clusters_names))
        logl_vals = rb.logl(df).tolist()
        logl.append(logl_vals)

    # Posterior for each row
    posterioris_x = []
    for row_idx in range(df.shape[0]):
        lklh = []
        for cluster_idx, _ in enumerate(clusters_names):
            x_val = math.exp(logl[cluster_idx][row_idx])
            lklh.append(x_val)
        t = sum(lklh)
        prob_posterior = [val / t for val in lklh]
        posterioris_x.append(prob_posterior)

    # sample times
    for s_i in range(sample):
        if s_i % 5 == 0:
            print(f"[DEBUG] structure_logl: sample iteration s_i={s_i}/{sample}")
        new_c = []
        df = dataframe.copy()
        for row_idx in range(df.shape[0]):
            new_c.append(np.random.choice(np.asarray(clusters_names), p=posterioris_x[row_idx]))

        df['cluster'] = pd.Series(pd.Categorical(new_c, categories=clusters_names))

        slogl.append(red.slogl(df))

    avg_slogl = sum(slogl) / len(slogl)
    print("[DEBUG] Leaving structure_logl(...)")
    return avg_slogl


def n_param(red, number_of_clusters, categories_df):
    print("[DEBUG] Entering n_param(...)")
    n = number_of_clusters - 1
    for var in red.children('cluster'):
        n += (red.num_parents(var)) * (len(categories_df[var]) - 1)
    print(f"[DEBUG] n_param returning {n}")
    return n


def sem(bn, dataframe, categories_df, clusters_names, max_iter=2, em_kmax=15, structlog_sample=15):
    print("[DEBUG] Entering sem(...)")
    print(f"[DEBUG] sem parameters: max_iter={max_iter}, em_kmax={em_kmax}, structlog_sample={structlog_sample}")
    print("[DEBUG] Running initial EM (Naive Bayes).")
    clgbn = EM(bn, dataframe, clusters_names, em_kmax)
    best = clgbn.clone()
    i = 0

    df = dataframe.copy()
    initial_logl = structure_logl(clgbn, df, clusters_names, structlog_sample)
    BIC = -2*initial_logl + math.log(df.shape[0]) * n_param(clgbn, len(clusters_names), categories_df)
    print(f"[DEBUG] Initial BIC={BIC}")

    participant_nodes = list(df.columns.copy())
    possible_arcs = list(itertools.permutations(participant_nodes, 2))
    print(f"[DEBUG] Number of possible arcs to check: {len(possible_arcs)}")

    while i < max_iter:
        print(f"[DEBUG] SEM outer loop iteration i={i}/{max_iter}. Current BIC={BIC}")
        s = 0
        k = 0
        random.shuffle(possible_arcs)

        # 1) Try adding arcs
        while k < len(possible_arcs):
            p_node, c_node = possible_arcs[k]
            if clgbn.can_add_arc(p_node, c_node):
                print(f"[DEBUG] Trying add_arc({p_node}->{c_node}). k={k}")
                red = pb.DiscreteBN(nodes=clgbn.nodes(), arcs=clgbn.arcs())
                red.add_arc(p_node, c_node)
                red = EM(red, df, clusters_names, em_kmax)
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

        # 2) Try flipping arcs
        k = 0
        possible = list(clgbn.arcs())
        for element in participant_nodes:
            if ('cluster', element) in possible:
                possible.remove(('cluster', element))
        random.shuffle(possible)

        while k < len(possible):
            p_node, c_node = possible[k]
            if clgbn.can_flip_arc(p_node, c_node):
                print(f"[DEBUG] Trying flip_arc({p_node}->{c_node}). k={k}")
                red = pb.DiscreteBN(nodes=clgbn.nodes(), arcs=clgbn.arcs())
                red.flip_arc(p_node, c_node)
                red = EM(red, df, clusters_names, em_kmax)
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

        # 3) Try removing arcs
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

    print("[DEBUG] Exiting sem(...). Returning best BN.")
    return best