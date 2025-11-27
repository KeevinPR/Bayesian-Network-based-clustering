import time
import pandas as pd
import pybnesian as pb
import discrete_structure
import discrete_analysis_hellinger
import os
import sys

def log_debug(msg):
    sys.stderr.write(f"[DEBUG {time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()

def run_benchmark():
    log_debug("Starting FULL benchmark...")
    
    # Load dataset
    csv_path = "datasets/customersSmall.csv"
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path, dtype=str)
    # Skip first col if it's ID (dash_clustering does this)
    df = df.iloc[:, 1:] 
    
    # Assuming standard format as per dash_clustering
    # Convert to category
    for col in df.columns:
        df[col] = df[col].astype(str).astype('category')
    
    # Strip column names
    df.columns = [c.strip() for c in df.columns]

    k_clusters = 2
    n_samples = 300 # Default we set in Dash
    
    cluster_names = [f'c{i}' for i in range(1, k_clusters + 1)]
    in_arcs = [('cluster', var) for var in df.columns]
    in_nodes = ['cluster'] + list(df.columns)
    bn_initial = pb.DiscreteBN(in_nodes, in_arcs)

    categories = {'cluster': cluster_names}
    for var in df.columns:
        categories[var] = df[var].cat.categories.tolist()

    # 1. Structure Learning (FULL MODE - mimicking Dash defaults)
    log_debug("Running Structure Learning (FULL MODE - OPTIMIZED)...")
    t0 = time.time()
    # Dash calls it without max_iter/em_kmax args, so it uses defaults: max_iter=2, em_kmax=50
    best_network = discrete_structure.sem(bn_initial, df, categories, cluster_names, max_iter=1, em_kmax=10)
    t_struct = time.time() - t0
    log_debug(f"[PROFILE] Structure learning took {t_struct:.4f} seconds")

    # 2. MAP Computation
    log_debug(f"Running MAP Computation with n={n_samples}...")
    t1 = time.time()
    map_reps = discrete_analysis_hellinger.get_MAP(best_network, cluster_names, n=n_samples)
    t_map = time.time() - t1
    log_debug(f"[PROFILE] MAP computation took {t_map:.4f} seconds")

    # 3. Importance Computation
    log_debug("Running Importance Computation...")
    ancestral_order = list(pb.Dag(best_network.nodes(), best_network.arcs()).topological_sort())
    if 'cluster' in ancestral_order:
        ancestral_order.remove('cluster')

    t2 = time.time()
    importances_dict = {}
    for clus in cluster_names:
        row = map_reps.loc[clus]
        point_list = []
        for var in ancestral_order:
            val = row[var]
            if isinstance(val, tuple):
                val = val[0]
            point_list.append(val)

        imp_clus = discrete_analysis_hellinger.importance_1(
            best_network, point_list, categories, cluster_names
        )
        importances_dict[clus] = imp_clus
    t_imp = time.time() - t2
    log_debug(f"[PROFILE] Importance computation took {t_imp:.4f} seconds")
    
    log_debug("Benchmark finished.")

if __name__ == "__main__":
    run_benchmark()
