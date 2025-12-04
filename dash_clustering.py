import dash
from dash import dcc, html, Input, Output, State, ALL
import dash_bootstrap_components as dbc

import base64
import io
import os
import matplotlib
matplotlib.use('Agg')  # For matplotlib with no display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx  # For DAG visualization

import pybnesian as pb  # or import pyAgrum, depending on your BN library
import time
import sys
def log_debug(msg):
    sys.stderr.write(f"[DEBUG {time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()
# === Import your code ===
import discrete_structure
import discrete_analysis_hellinger
import discrete_representation
import pybnesianCPT_to_df


app = dash.Dash(
    __name__,
    
    #Optional
    requests_pathname_prefix='/Model/LearningFromData/ClusteringDash/',
    
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)

server = app.server

############################################################
# HELPER: Make a DAG figure from a learned BN using networkx
############################################################
def plot_bn_dag(bn: pb.DiscreteBN, title="Learned BN Structure"):
    print("[DEBUG] Entering plot_bn_dag...")
    fig, ax = plt.subplots(figsize=(6, 4))
    G = nx.DiGraph()
    G.add_nodes_from(bn.nodes())
    G.add_edges_from(bn.arcs())

    pos = nx.spring_layout(G, seed=42)  # or any layout you prefer
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='skyblue', node_size=800, edgecolors='k')
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
    nx.draw_networkx_edges(
        G,
        pos,
        ax=ax,
        arrows=True,               # ensure arrow heads
        arrowstyle='-|>',          # a more pointy arrow style
        arrowsize=20,              # larger arrow head
        min_source_margin=10,      # space from node boundary
        min_target_margin=10
    )

    ax.set_title(title, fontsize=12)
    ax.set_axis_off()

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    print("[DEBUG] Leaving plot_bn_dag.")
    return f"data:image/png;base64,{encoded}"


############################################################
# HELPER: Produce a radar chart with MAP reps + importances
############################################################
def plot_map_with_importance(maps_df, importances, df_categories):
    """
    maps_df: Pandas DataFrame of shape (#clusters, #variables)
             Each cell is a tuple (category, probability)
             or just a single category if you used get_MAP_simple.
    importances: dict of {cluster_name: {variable: importance_value}}
    df_categories: from df_to_dict, e.g., discrete_analysis_hellinger.df_to_dict(df)
    """
    from radar_chart_discrete_categories import ComplexRadar
    import io, base64
    import matplotlib.pyplot as plt

    #fig = plt.figure(figsize=(8, 8))
    fig = plt.figure(figsize=(8, 8), dpi=120)
    variables = list(maps_df.columns)  # e.g. ['Age', 'Sex', 'Income', ...]
    cluster_list = list(maps_df.index) # e.g. ['c1','c2']

    # 1) Instantiate the radar passing the dictionary df_categories as second arg
    radar = ComplexRadar(fig, df_categories, show_scales=True)

    # 2) For each cluster (row in maps_df), build a dict {var -> chosen_category}
    for clus in cluster_list:
        row_dict = {}
        for var in variables:
            item = maps_df.loc[clus, var]  
            # item might be a tuple (chosen_cat, probability) or just a category string
            if isinstance(item, tuple):
                chosen_cat = item[0]
            else:
                chosen_cat = item

            row_dict[var] = chosen_cat
        
        label_text = f"Cluster {clus}"
        # Now we pass row_dict to .plot(...) and .fill(...)
        # so that `_scale_data` can do df_categories[var][data[var]]
        radar.plot(row_dict, df_categories, label=label_text)
        radar.fill(row_dict, df_categories, alpha=0.2)

    radar.set_title("MAP + Importance Radar Chart", fontsize=12, pad=40)
    radar.use_legend(loc='upper right')

    # 3) Convert to base64
    buf = io.BytesIO()
    #fig.savefig(buf, format='png')
    fig.savefig(buf, format='png', bbox_inches='tight') #Trims extra margins around the final image
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)

    return f"data:image/png;base64,{encoded}"



# ====================== LAYOUT ======================
app.layout = dcc.Loading(
    id="global-spinner",
    overlay_style={"visibility":"visible", "filter":"blur(1px)"},
    type="circle",
    fullscreen=False,
    children=html.Div([
        html.H1("Bayesian Network Clustering", style={'textAlign': 'center'}),

        # Upload section
        html.Div([
            html.H3("Upload Discrete Dataset (CSV)"),
            dcc.Upload(
                id='upload-data',
                children=html.Button('Upload File (CSV)'),
                multiple=False
            ),
            # Checklist to use a default dataset
            dcc.Checklist(
                id='use-default-dataset',
                options=[{'label': 'Use default dataset', 'value': 'default'}],
                value=[],
                style={'textAlign': 'center', 'marginTop': '10px'}
            ),
            html.Div(id='upload-data-filename', style={'marginTop': '5px', 'color': 'green'})
        ], style={'textAlign': 'center'}),

        html.Hr(),

        # Choose Action
        html.Div([
            html.H3("Choose Action"),
            dcc.RadioItems(
                id='action-radio',
                options=[
                    {'label': 'Cluster Network Only', 'value': 'cluster_only'},
                    {'label': 'Cluster Network + Importance Analysis', 'value': 'cluster_importance'}
                ],
                value='cluster_only',
                style={'textAlign': 'center'}
            )
        ], style={'textAlign': 'center'}),

        html.Br(),

        # cluster_only params
        html.Div([
            html.Label("Number of Clusters:"),
            dcc.Input(
                id='num-clusters-input',
                type='number',
                value=2,
                min=2,
                step=1
            ),
            html.Button("Run Clustering", id='run-clustering-button', n_clicks=0)
        ], id='cluster-only-params', style={'textAlign': 'center', 'display': 'block'}),

        # cluster_importance params
        html.Div([
            html.Label("Number of Clusters:"),
            dcc.Input(
                id='num-clusters-importance-input',
                type='number',
                value=2,
                min=2,
                step=1,
                style={'marginRight': '20px'}
            ),
            html.Label("Sample for Inference:"),
            dcc.Input(
                id='num-samples-input',
                type='number',
                value=50,
                min=10,
                step=10,
                style={'marginRight': '20px'}
            ),
            html.Small("Recomendado 30-100 muestras (~9x7 / 500 filas). Más muestras = más lento.",
                       style={'display': 'block', 'marginTop': '5px', 'color': '#555'}),
            html.Br(),
            html.Label("Order Variables?"),
            dcc.RadioItems(
                id='variable-order-radio',
                options=[
                    {'label': 'Random Category Order', 'value': 'random'},
                    {'label': 'Manually Select Variable Order', 'value': 'manual'},
                    {'label': 'Skip Order Selection', 'value': 'skip'}
                ],
                value='skip',
                style={'marginTop': '10px'}
            ),
            html.Br(),
            html.Div(id='manual-order-container', style={'display': 'none'}),
            html.Button("Continue", id='continue-importance-button', n_clicks=0, style={'marginTop': '10px'}),
            html.Br(),
            html.Button(
                "Run Clustering + Importance",
                id='run-clustering-importance-button',
                n_clicks=0,
                style={'display': 'none', 'marginTop': '10px'}
            )
        ], id='cluster-importance-params', style={'textAlign': 'center', 'display': 'none'}),

        html.Hr(),

        # Output area
        html.Div(id='output-area', style={'textAlign': 'center'}),

        # dcc.Stores
        dcc.Store(id='stored-data'),
        dcc.Store(id='stored-dataframe')
    ])
)


# ====================== CALLBACKS ======================

@app.callback(
    [Output('cluster-only-params', 'style'),
     Output('cluster-importance-params', 'style')],
    Input('action-radio', 'value')
)
def toggle_param_sections(action_value):
    print(f"[DEBUG] toggle_param_sections called with action_value={action_value}")
    if action_value == 'cluster_only':
        return ({'textAlign': 'center', 'display': 'block'},
                {'textAlign': 'center', 'display': 'none'})
    else:
        return ({'textAlign': 'center', 'display': 'none'},
                {'textAlign': 'center', 'display': 'block'})


# Handle upload or default dataset
@app.callback(
    Output('upload-data-filename', 'children'),
    Output('stored-data', 'data'),
    Input('upload-data', 'contents'),
    Input('use-default-dataset', 'value'),
    State('upload-data', 'filename')
)
def handle_upload_or_default(uploaded_contents, default_value, uploaded_filename):
    """
    If the user checks 'default', read from a local CSV path.
    Otherwise, if the user has uploaded a file, parse that file.
    """
    print("[DEBUG] Entering handle_upload_or_default callback...")
    default_csv_path = "/var/www/html/CIGModels/backend/cigmodelsdjango/cigmodelsdjangoapp/Bayesian-Network-based-clustering/datasets/customersSmall.csv"

    # If 'default' is selected
    if 'default' in default_value:
        print("[DEBUG] 'default' in default_value -> using default dataset")
        try:
            df_default = pd.read_csv(default_csv_path, dtype=str)
            raw_text = df_default.to_csv(index=False)
            msg = f"Using default dataset: {os.path.basename(default_csv_path)}"
            print("[DEBUG] Successfully read default dataset.")
            return msg, raw_text
        except Exception as e:
            print(f"[DEBUG] Error reading default dataset: {str(e)}")
            return f"Error reading default dataset: {str(e)}", None

    # Otherwise, handle user upload
    if uploaded_contents is not None:
        print("[DEBUG] A file was uploaded by the user.")
        content_type, content_string = uploaded_contents.split(',')
        decoded = base64.b64decode(content_string)
        msg = f"File uploaded: {uploaded_filename}"
        print("[DEBUG] File decoded successfully.")
        return msg, decoded.decode('utf-8')

    print("[DEBUG] No default selected, no file uploaded -> PreventUpdate")
    raise dash.exceptions.PreventUpdate


@app.callback(
    Output('stored-dataframe', 'data'),
    Input('stored-data', 'data')
)
def parse_to_dataframe(raw_text):
    print("[DEBUG] parse_to_dataframe called.")
    if raw_text is None:
        print("[DEBUG] No raw_text -> PreventUpdate")
        raise dash.exceptions.PreventUpdate

    import io
    print("[DEBUG] Converting raw_text to DataFrame...")
    df = pd.read_csv(io.StringIO(raw_text), dtype=str)
    
    # Skip the first column in case it's an ID or something.
    # This slices rows [:], columns [1:] => keep all rows, drop col index 0
    df = df.iloc[:, 1:]  

    # Convert every column to category
    for col in df.columns:
        df[col] = df[col].astype(str).astype('category')
    
    # Strip column names
    df.columns = [c.strip() for c in df.columns]

    # Convert every column to category
    for col in df.columns:
        df[col] = df[col].astype(str).astype('category')

    print(f"[DEBUG] DataFrame shape: {df.shape}, columns: {list(df.columns)}")
    return df.to_json(date_format='iso', orient='split')


@app.callback(
    [Output('manual-order-container', 'children'),
     Output('manual-order-container', 'style')],
    Input('variable-order-radio', 'value'),
    State('stored-dataframe', 'data')
)
def show_manual_order(order_choice, df_json):
    print("[DEBUG] show_manual_order called.")
    if not df_json or order_choice != 'manual':
        print("[DEBUG] Not in 'manual' mode or no df_json. Hiding manual-order-container.")
        return ([], {'display': 'none'})

    df = pd.read_json(df_json, orient='split')
    var_list = df.columns.tolist()
    print(f"[DEBUG] Building manual order for columns: {var_list}")
    children = []
    used_positions = list(range(1, len(var_list) + 1))
    for var in var_list:
        children.append(
            html.Div([
                html.Label(f"Order for variable {var}"),
                dcc.Dropdown(
                    id={'type': 'var-order-dropdown', 'index': var},
                    options=[{'label': str(pos), 'value': pos} for pos in used_positions],
                    placeholder='Select position'
                )
            ], style={'marginBottom': '10px'})
        )
    return (children, {'display': 'block'})


@app.callback(
    Output('run-clustering-importance-button', 'style'),
    Input('continue-importance-button', 'n_clicks'),
    State('variable-order-radio', 'value'),
    State({'type': 'var-order-dropdown', 'index': ALL}, 'value')
)
def show_final_run_button(n_clicks, order_choice, all_values):
    print("[DEBUG] show_final_run_button called.")
    if n_clicks is None or n_clicks == 0:
        print("[DEBUG] No clicks on continue -> hide the button.")
        return {'display': 'none'}

    if order_choice == 'manual':
        # Check if any position is None or duplicates
        if None in all_values or len(set(all_values)) != len(all_values):
            print("[DEBUG] Manual order incomplete or duplicates -> hide the button.")
            return {'display': 'none'}

    print("[DEBUG] Continue clicked, showing the final run button.")
    return {'display': 'inline-block'}


@app.callback(
    Output('output-area', 'children', allow_duplicate=True),
    Input('run-clustering-importance-button', 'n_clicks'),
    State('num-clusters-importance-input', 'value'),
    State('num-samples-input', 'value'),
    State('variable-order-radio', 'value'),
    State({'type': 'var-order-dropdown', 'index': ALL}, 'value'),
    State({'type': 'var-order-dropdown', 'index': ALL}, 'id'),
    State('stored-dataframe', 'data'),
    prevent_initial_call='initial_duplicate'
)
def run_cluster_importance(
    n_clicks, k_clusters, n_samples, order_choice,
    all_values, all_ids, df_json
):
    """
    Runs clustering + importance analysis, then displays:
     1) A single BN figure,
     2) A carousel with subcluster DAGs,
     3) The radar chart for MAP representatives + importance,
     4) A textual list of each (variable, map_value, importance).
    """
    with open("/tmp/dash_debug.log", "a") as f:
        f.write(f"\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] === NEW RUN ===\n")
        f.write(f"Parameters: k_clusters={k_clusters}, n_samples={n_samples}\n")
    log_debug("run_cluster_importance called.")
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not df_json:
        return "No dataset found. Please upload a CSV or use the default dataset."
    
    # 1) Prepare DataFrame
    df = pd.read_json(df_json, orient='split')
    with open("/tmp/dash_debug.log", "a") as f:
        f.write(f"Dataset shape: {df.shape} (rows={df.shape[0]}, cols={df.shape[1]})\n")
    for col in df.columns:
        df[col] = df[col].astype(str).astype('category')

    # 2) Possibly reorder columns if manual
    if order_choice == 'manual':
        print("[DEBUG] Reordering columns manually.")
        var_positions = {}
        for val, id_ in zip(all_values, all_ids):
            var_positions[id_['index']] = val
        sorted_vars = sorted(var_positions.items(), key=lambda x: x[1])
        new_col_order = [s[0] for s in sorted_vars]
        df = df[new_col_order]
    else:
        print("[DEBUG] Keeping original column order (or random/skip).")

    # 3) Build BN structure
    cluster_names = [f'c{i}' for i in range(1, k_clusters + 1)]
    in_arcs = [('cluster', var) for var in df.columns]
    in_nodes = ['cluster'] + list(df.columns)
    bn_initial = pb.DiscreteBN(in_nodes, in_arcs)

    categories = {'cluster': cluster_names}
    for var in df.columns:
        categories[var] = df[var].cat.categories.tolist()

    # 4) Learn the BN
    log_debug("Calling discrete_structure.sem(...)")
    t0 = time.time()
    # Use max_iter=1, em_kmax=10 to speed up structure learning
    best_network = discrete_structure.sem(bn_initial, df, categories, cluster_names, max_iter=1, em_kmax=3)
    elapsed = time.time() - t0
    with open("/tmp/dash_debug.log", "a") as f:
        f.write(f"Structure learning: {elapsed:.2f}s\n")
    log_debug(f"Structure learning took {elapsed} seconds")

    # 5) Single BN figure
    print("[DEBUG] Plotting the main BN DAG.")
    dag_img_src = plot_bn_dag(best_network, "Cluster + Importance BN")

    # 6) Obtain MAP representatives + importance
    # 6) Obtain MAP representatives + importance
    log_debug(f"Calling get_MAP(...) with n={n_samples}")
    t1 = time.time()
    map_reps = discrete_analysis_hellinger.get_MAP(best_network, cluster_names, n=n_samples)
    elapsed = time.time() - t1
    with open("/tmp/dash_debug.log", "a") as f:
        f.write(f"MAP computation: {elapsed:.2f}s\n")
    log_debug(f"MAP computation took {elapsed} seconds")
    
    # Prepare an ancestral order that excludes 'cluster' itself
    ancestral_order = list(pb.Dag(best_network.nodes(), best_network.arcs()).topological_sort())
    if 'cluster' in ancestral_order:
        ancestral_order.remove('cluster')

    # Compute importance for each cluster
    t2 = time.time()
    importances_dict = {}
    for clus in cluster_names:
        row = map_reps.loc[clus]
        point_list = []
        for var in ancestral_order:
            val = row[var]
            # If row[var] is a tuple (cat, prob), just keep the category
            if isinstance(val, tuple):
                val = val[0]
            point_list.append(val)

        imp_clus = discrete_analysis_hellinger.importance_1(
            best_network, point_list, categories, cluster_names
        )
        importances_dict[clus] = imp_clus
    elapsed = time.time() - t2
    with open("/tmp/dash_debug.log", "a") as f:
        f.write(f"Importance computation: {elapsed:.2f}s\n")
        f.write(f"TOTAL TIME: {time.time() - t0:.2f}s\n")
    log_debug(f"Importance computation took {elapsed} seconds")
    
    # 7) Build the subcluster DAGs + carousel
    print("[DEBUG] Building subcluster DAGs for each cluster.")
    cluster_images_list = clusters_dags_as_base64(best_network, importances_dict, cluster_names)

    import dash_bootstrap_components as dbc
    items = []
    for cname, img_src in zip(cluster_names, cluster_images_list):
        items.append({
            "key": cname,
            "src": img_src,
            "header": f"Cluster {cname}",
            "caption": f"Subcluster DAG for {cname}"
        })
    carousel = dbc.Carousel(
        items=items,
        controls=True,
        indicators=False,
        interval=None,
        ride="carousel",
        style={"maxWidth": "600px", "margin": "0 auto"}
    )

    # 8) Plot the radar chart
    print("[DEBUG] Plotting radar chart with MAP + importance.")
    df_categories = discrete_analysis_hellinger.df_to_dict(df)
    radar_img_src = plot_map_with_importance(map_reps, importances_dict, df_categories)

    arcs_list = list(best_network.arcs())
   
    #Removing arcs showing 
    #print(f"[DEBUG] Learned BN arcs: {arcs_list}")

    # 9) Build a textual display of (var, map_value) + importance for each cluster
    print("[DEBUG] Building textual display of MAP values + importance.")

    # We'll collect each cluster's info in a list of Divs, each displayed as a column:
    cluster_columns = []
    for clus in cluster_names:
        row = map_reps.loc[clus]
        
        # Build lines for each variable, but now as plain text in small Divs—no bullets:
        lines = []
        for var in ancestral_order:
            chosen_val = row[var]
            if isinstance(chosen_val, tuple):
                chosen_val = chosen_val[0]
            imp_val = importances_dict[clus].get(var, 0.0)
            line_str = f"({var}, {chosen_val}) importance {imp_val:.4f}"
            lines.append(
                html.Div(line_str, style={'fontSize': '12px', 'marginBottom': '3px'})
            )

        # Wrap them in one Div for this cluster, with inline-block to place columns side by side:
        cluster_div = html.Div(
            children=[
                html.H5(f"cluster {clus}", style={'fontSize': '14px', 'marginBottom': '8px'}),
                *lines
            ],
            style={
                'display': 'inline-block',
                'verticalAlign': 'top',
                'width': '250px',    # Adjust column width as needed
                'margin': '10px',    # Spacing between columns
                'fontFamily': 'Arial, sans-serif',
                'border': '1px solid #ccc',  # optional border
                'padding': '8px'
            }
        )
        cluster_columns.append(cluster_div)

    # A flex container for all columns:
    map_importances_container = html.Div(
        children=cluster_columns,
        style={
            'display': 'flex',
            'flexWrap': 'wrap',
            'justifyContent': 'center',
            'alignItems': 'flex-start',
            'marginTop': '20px'
        }
    )

    # 10) Build final layout
    layout_div = html.Div([
        html.H4("Clustering + Importance Analysis Results"),
        html.P(f"Number of clusters = {k_clusters}"),
        html.P(f"Samples for inference = {n_samples}"),
        html.Hr(),

        # Single BN figure
        html.Img(
            src=dag_img_src,
            className="zoomable",
            style={'maxWidth': '600px', 'display': 'block', 'margin': '0 auto'}
        ),

        html.Hr(),
        html.H4("Individual Subcluster DAGs (Carousel)"),
        carousel,

        html.Hr(),
        html.H5("Radar Chart of MAP Representatives + Importance"),
        html.Img(
            src=radar_img_src,
            className="zoomable",
            style={'maxWidth': '600px', 'display': 'block', 'margin': '0 auto'}
        ),

        # Here is where we show the textual lines for each cluster
        html.Div([
            html.H5("MAP + Importance by Cluster", style={'marginTop': '20px'}),
                map_importances_container
            ])
    ])
    print("[DEBUG] run_cluster_importance returning layout_div.")
    return layout_div


from discrete_representation import clusters_dags_as_base64
import dash_bootstrap_components as dbc

@app.callback(
    Output('output-area', 'children'),
    Input('run-clustering-button', 'n_clicks'),
    State('num-clusters-input', 'value'),
    State('stored-dataframe', 'data'),
    prevent_initial_call=True
)
def run_cluster_only(n_clicks, k_clusters, df_json):
    # 1) Check for dataset availability
    if not n_clicks:
        raise dash.exceptions.PreventUpdate
    if not df_json:
        return "No dataset found. Please upload a CSV or use the default dataset."
    

    # 2) Prepare the DataFrame
    df = pd.read_json(df_json, orient='split')
    for col in df.columns:
        df[col] = df[col].astype(str).astype('category')

    # 3) Build the BN
    cluster_names = [f'c{i}' for i in range(1, k_clusters + 1)]
    in_arcs = [('cluster', col) for col in df.columns]
    in_nodes = ['cluster'] + list(df.columns)
    bn_initial = pb.DiscreteBN(in_nodes, in_arcs)

    # 4) Categories dictionary
    categories = {'cluster': cluster_names}
    for var in df.columns:
        categories[var] = df[var].cat.categories.tolist()

    # 5) Train/learn the BN structure
    # Use max_iter=1, em_kmax=10 to speed up structure learning
    best_network = discrete_structure.sem(bn_initial, df, categories, cluster_names, max_iter=1, em_kmax=3)

    # 6) Single BN figure
    single_bn_src = plot_bn_dag(best_network, "Learned BN (Cluster Only)")

    # 7) Dummy importance (or real) for each cluster
    dummy_importance = {}
    for c in cluster_names:
        dummy_importance[c] = {}
        for node in best_network.nodes():
            dummy_importance[c][node] = np.random.random()

    # 8) One DAG image per cluster
    cluster_images_list = clusters_dags_as_base64(best_network, dummy_importance, cluster_names)

    # 9) Build `items` list for dbc.Carousel
    items = []
    for cname, img_src in zip(cluster_names, cluster_images_list):
        items.append({
            "key": cname,          # unique key
            "src": img_src,        # base64 image as "src"
            "header": f"Cluster {cname}",
            "caption": f"Subcluster DAG for {cname}"
        })
    
    # Create a Carousel component
    carousel = dbc.Carousel(
        items=items,
        controls=True,        # show left/right arrows
        indicators=False,      # show clickable indicators at bottom
        interval=None,        # set to None to disable auto-play
        ride="carousel",      # or remove "ride" for a static carousel
        style={"maxWidth": "600px", "margin": "0 auto"},
        #className="clustering-carousel"
    )

    arcs_list = list(best_network.arcs())

    # 10) Return layout with single BN and a carousel for subclusters
    return html.Div([
        html.H4("Clustering Completed (Network Only)"),
        html.P(f"Number of clusters = {k_clusters}"),

        #List of arcs
        #html.H5("Arcs:"),
        #html.Ul(
        #    [html.Li(str(arc), style={'fontSize': '14px', 'marginBottom': '5px'}) for arc in arcs_list],
        #    style={
        #        'listStyleType': 'none',
        #        'paddingLeft': '20px',
        #        'margin': '0'
        #    }
        #),

        # Single BN figure
        html.Img(src=single_bn_src, className="zoomable", style={'maxWidth': '600px'}),

        html.Hr(),
        html.H4("Individual Subcluster DAGs (Carousel)"),
        
        # The carousel goes here:
        carousel
    ])
    
if __name__ == '__main__':
    print("[DEBUG] Starting Dash server on port 8055...")
    #Default port is 8050
    app.run_server(debug=True, host='0.0.0.0', port=8055)