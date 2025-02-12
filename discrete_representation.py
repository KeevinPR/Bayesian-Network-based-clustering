import networkx as nx
import pandas as pd
import plotly
import pickle
import pybnesian as pb
import numpy as np
import matplotlib.pyplot as plt

#Este código genera una representación de la red con el sombreado de nodos según los valores de importancia introducidos.
#Esto se utiliza para representar la red dado el representante de cada cluster y su importancia

def clusters_dag(red,importance,clusters_names):
    fig1,axes=plt.subplots(1,len(clusters_names))
    for i in range(len(clusters_names)):
        G = nx.DiGraph()
        G.add_nodes_from(red.nodes())
        G.add_edges_from(red.arcs())
        values = [importance[clusters_names[i]].get(node, 0) for node in G.nodes()]

        #pos = nx.planar_layout(G)
        pos = nx.bipartite_layout(G,['cluster'])
        nx.draw_networkx_nodes(G, pos, cmap=plt.get_cmap('Purples'),ax=axes[i],
                               node_color=values, node_size=2000, vmin=min(values), vmax=max(values), edgecolors='k')
        nx.draw_networkx_labels(G, pos, font_size=12,ax=axes[i])
        nx.draw_networkx_edges(G, pos, arrows=True, node_size=2000,ax=axes[i],connectionstyle='arc3,rad=-0.3')

        axes[i].set_title('cluster'+' '+f"{clusters_names[i]}")

    plt.show()
    
import io, base64
def clusters_dags_as_base64(bn, importance, cluster_names):
    """
    Generate an image for each cluster and return
    a list of base64-encoded PNG strings (one per cluster).
    """
    encoded_images = []
    for cluster in cluster_names:
        fig, ax = plt.subplots(figsize=(8, 6))

        # Build the graph from the BN
        G = nx.DiGraph()
        G.add_nodes_from(bn.nodes())
        G.add_edges_from(bn.arcs())

        # Get importance values for this cluster
        values = [importance[cluster].get(node, 0) for node in G.nodes()]

        # Use bipartite_layout or any other layout you prefer
        pos = nx.bipartite_layout(G, ['cluster'])

        # Draw nodes, edges, and labels
        nx.draw_networkx_nodes(
            G, pos, cmap=plt.get_cmap('Purples'),
            node_color=values, node_size=2000, edgecolors='k', ax=ax
        )
        nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)
        nx.draw_networkx_edges(
            G, pos, arrows=True, node_size=2000, ax=ax,
            connectionstyle='arc3,rad=-0.3'
        )

        ax.set_title(f"Cluster {cluster}", fontsize=14)
        ax.set_axis_off()
        fig.tight_layout()

        # Save figure to buffer to avoid partial cropping
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)

        # Convert buffer to base64 string
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        encoded_images.append("data:image/png;base64," + encoded)

        plt.close(fig)

    return encoded_images