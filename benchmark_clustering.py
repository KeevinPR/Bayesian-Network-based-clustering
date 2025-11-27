import time
import pandas as pd
import pybnesian as pb
import discrete_structure
import discrete_analysis_hellinger
import os

def run_benchmark():
    print("Starting benchmark...")
    
    # Load dataset
    csv_path = "datasets/customersSmall.csv"
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path, dtype=str)
    # Skip first col if it's ID (dash_clustering does this)
    df = df.iloc[:, 1:] 
    # Wait, dash_clustering.py says:
    # df = df.iloc[:, 1:] 
    # Let's check the file content first to be sure.
    
    # Assuming standard format as per dash_clustering
    # Convert to category
    for col in df.columns:
        df[col] = df[col].astype(str).astype('category')
    
    # Strip column names
    df.columns = [c.strip() for c in df.columns]

    k_clusters = 2
    n_samples = 300 # Default we set
    
    cluster_names = [f'c{i}' for i in range(1, k_clusters + 1)]
    in_arcs = [('cluster', var) for var in df.columns]
    in_nodes = ['cluster'] + list(df.columns)
    bn_initial = pb.DiscreteBN(in_nodes, in_arcs)

    categories = {'cluster': cluster_names}
    for var in df.columns:
        categories[var] = df[var].cat.categories.tolist()

    # 1. Structure Learning
    print("Running Structure Learning (Fast Mode)...")
    t0 = time.time()
    # Use max_iter=0, em_kmax=1 to skip heavy optimization and just get a result quickly
    best_network = discrete_structure.sem(bn_initial, df, categories, cluster_names, max_iter=0, em_kmax=1)
    t_struct = time.time() - t0
    print(f"[PROFILE] Structure learning took {t_struct:.4f} seconds")

    # 2. MAP Computation
    print("Running MAP Computation...")
    t1 = time.time()
    map_reps = discrete_analysis_hellinger.get_MAP(best_network, cluster_names, n=n_samples)
    t_map = time.time() - t1
    print(f"[PROFILE] MAP computation took {t_map:.4f} seconds")

    # 3. Importance Computation
    print("Running Importance Computation...")
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
    print(f"[PROFILE] Importance computation took {t_imp:.4f} seconds")
    
    print("Benchmark finished.")

if __name__ == "__main__":
    run_benchmark()
