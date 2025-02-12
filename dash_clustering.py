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
# === Import your code ===
import discrete_structure
import discrete_analysis_hellinger
import discrete_representation
import pybnesianCPT_to_df


app = dash.Dash(
    __name__,
    requests_pathname_prefix='/Model/LearningFromData/ClusteringDash/',
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)


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
    nx.draw_networkx_edges(G, pos, ax=ax, arrowstyle='->', arrowsize=10)

    ax.set_title(title, fontsize=12)
    ax.set_axis_off()

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format='png')
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
    print("[DEBUG] Entering plot_map_with_importance...")
    from radar_chart_discrete_categories import ComplexRadar

    fig = plt.figure(figsize=(8, 8))
    variables = list(maps_df.columns)
    cluster_list = list(maps_df.index)

    # Build min/max bounds for each variable based on how many categories exist
    bounds = []
    for var in variables:
        cat_count = len(df_categories[var])
        bounds.append([1, cat_count])  # categories start at 1 in your code

    radar = ComplexRadar(fig, variables, bounds, show_scales=True)

    for clus in cluster_list:
        row_values = []
        for var in variables:
            item = maps_df.loc[clus, var]
            if isinstance(item, tuple):
                chosen_cat = item[0]
            else:
                chosen_cat = item

            code = df_categories[var][chosen_cat]
            row_values.append(code)

        label_text = f"Cluster {clus}"
        radar.plot(row_values, df_categories, label=label_text)
        radar.fill(row_values, df_categories, alpha=0.2)

    radar.set_title("MAP + Importance Radar Chart", fontsize=12)
    radar.use_legend(loc='upper right')

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    print("[DEBUG] Leaving plot_map_with_importance.")
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
                value=100000,
                min=1000,
                step=1000,
                style={'marginRight': '20px'}
            ),
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
        html.Div(id='output-area-two', style={'textAlign': 'center'}),

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
    Input('run-clustering-button', 'n_clicks'),
    State('num-clusters-input', 'value'),
    State('stored-dataframe', 'data'),
    prevent_initial_call='initial_duplicate'
)
def run_cluster_only(n_clicks, k_clusters, df_json):
    print(f"[DEBUG] run_cluster_only called with n_clicks={n_clicks}, k_clusters={k_clusters}")
    if not df_json:
        print("[DEBUG] No dataset in stored-dataframe -> returning message.")
        return "No dataset found. Please upload a CSV or use the default dataset."
    if n_clicks is None or n_clicks == 0:
        print("[DEBUG] No clicks -> PreventUpdate.")
        raise dash.exceptions.PreventUpdate

    df = pd.read_json(df_json, orient='split')

    # Convert columns to category again
    for col in df.columns:
        df[col] = df[col].astype(str).astype('category')

    print(f"[DEBUG] Building naive BN for cluster-only. DataFrame shape={df.shape}")
    in_arcs = [('cluster', var) for var in df.columns]
    in_nodes = ['cluster'] + list(df.columns)
    cluster_names = [f'c{i}' for i in range(1, k_clusters + 1)]

    red_inicial = pb.DiscreteBN(in_nodes, in_arcs)

    # Build categories dictionary
    categories = {'cluster': cluster_names}
    for var in df.columns:
        categories[var] = df[var].cat.categories.tolist()

    print("[DEBUG] Calling discrete_structure.sem(...)")
    best_network = discrete_structure.sem(red_inicial, df, categories, cluster_names)
    arcs_list = list(best_network.arcs())
    print(f"[DEBUG] Learned BN arcs: {arcs_list}")

    dag_img_src = plot_bn_dag(best_network, "Learned BN (Cluster Only)")
    result_divs = [
        html.H4("Clustering Completed (Network Only)"),
        html.P(f"Number of clusters = {k_clusters}"),
        html.H5("Arcs:"),
        html.Ul([html.Li(str(arc)) for arc in arcs_list]),
        html.Img(src=dag_img_src, style={'maxWidth': '500px'})
    ]
    return result_divs


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
    print("[DEBUG] run_cluster_importance called.")
    print(f"[DEBUG] n_clicks={n_clicks}, k_clusters={k_clusters}, n_samples={n_samples}, order_choice={order_choice}")
    if not df_json:
        print("[DEBUG] No dataset in stored-dataframe -> returning message.")
        return "No dataset found. Please upload a CSV or use the default dataset."
    if n_clicks is None or n_clicks == 0:
        print("[DEBUG] No clicks -> PreventUpdate.")
        raise dash.exceptions.PreventUpdate

    df = pd.read_json(df_json, orient='split')
    
    for col in df.columns:
        df[col] = df[col].astype(str).astype('category')

    # Reorder columns if manual
    if order_choice == 'manual':
        print("[DEBUG] Reordering columns manually.")
        var_positions = {}
        for val, id_ in zip(all_values, all_ids):
            var_positions[id_['index']] = val
        sorted_vars = sorted(var_positions.items(), key=lambda x: x[1])
        new_col_order = [s[0] for s in sorted_vars]
        df = df[new_col_order]
    else:
        print("[DEBUG] Keeping original column order or random/skip logic outside of manual mode.")

    print("[DEBUG] Building naive BN structure for cluster + importance.")
    in_arcs = [('cluster', var) for var in df.columns]
    in_nodes = ['cluster'] + list(df.columns)
    cluster_names = [f'c{i}' for i in range(1, k_clusters + 1)]
    red_inicial = pb.DiscreteBN(in_nodes, in_arcs)

    categories = {'cluster': cluster_names}
    for var in df.columns:
        categories[var] = df[var].cat.categories.tolist()

    print("[DEBUG] Calling discrete_structure.sem(...)")
    best_network = discrete_structure.sem(red_inicial, df, categories, cluster_names)

    print("[DEBUG] Calling get_MAP(...)")
    map_reps = discrete_analysis_hellinger.get_MAP(best_network, cluster_names, n=n_samples)

    print("[DEBUG] Computing importance for each cluster.")
    ancestral_order = list(pb.Dag(best_network.nodes(), best_network.arcs()).topological_sort())
    if 'cluster' in ancestral_order:
        ancestral_order.remove('cluster')

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

    print("[DEBUG] Building df_categories for radar chart.")
    df_categories = discrete_analysis_hellinger.df_to_dict(df)

    print("[DEBUG] Plotting radar chart + DAG.")
    radar_img_src = plot_map_with_importance(map_reps, importances_dict, df_categories)
    dag_img_src = plot_bn_dag(best_network, "Cluster + Importance BN")

    arcs_list = list(best_network.arcs())
    print(f"[DEBUG] Learned BN arcs (cluster+importance): {arcs_list}")

    layout_div = html.Div([
        html.H4("Clustering + Importance Analysis Results"),
        html.P(f"Number of clusters = {k_clusters}"),
        html.P(f"Samples for inference = {n_samples}"),
        html.Hr(),
        html.H5("Arcs in the Learned BN:"),
        html.Ul([html.Li(str(arc)) for arc in arcs_list]),
        html.Img(src=dag_img_src, style={'maxWidth': '400px', 'display': 'block', 'margin': '0 auto'}),
        html.Hr(),
        html.H5("Radar Chart of MAP Representatives + Importance"),
        html.Img(src=radar_img_src, style={'maxWidth': '500px', 'display': 'block', 'margin': '0 auto'})
    ])
    print("[DEBUG] run_cluster_importance returning layout_div.")
    return layout_div

from discrete_representation import clusters_dag_as_base64

@app.callback(
    Output('output-area-two', 'children'),
    Input('run-clustering-button', 'n_clicks'),
    State('num-clusters-input', 'value'),
    State('stored-dataframe', 'data'),
    prevent_initial_call=True
)
def run_cluster_only(n_clicks, k_clusters, df_json):
    if not df_json:
        return "No dataset found. Please upload a CSV or use the default dataset."
    if not n_clicks:
        raise dash.exceptions.PreventUpdate

    # 1) Prepare your dataframe
    df = pd.read_json(df_json, orient='split')
    for col in df.columns:
        df[col] = df[col].astype(str).astype('category')

    # 2) Build BN
    cluster_names = [f'c{i}' for i in range(1, k_clusters+1)]
    in_arcs = [('cluster', col) for col in df.columns]
    in_nodes = ['cluster'] + list(df.columns)
    bn_initial = pb.DiscreteBN(in_nodes, in_arcs)

    # categories dict
    categories = {'cluster': cluster_names}
    for var in df.columns:
        categories[var] = df[var].cat.categories.tolist()

    # 3) Call your SEM
    best_network = discrete_structure.sem(bn_initial, df, categories, cluster_names)
    
    # 4) Convert arcs to an image
    single_bn_src = plot_bn_dag(best_network, "Learned BN (Cluster Only)")

    # 5) If you want "clusters_dag" style subplots, we need an 'importance' dict
    #    If you are ignoring importance, just feed zeros. E.g.:
    dummy_importance = {}
    for c in cluster_names:
        # For each cluster, we have a dict of node->value. We'll just do random or all zero
        # random approach:
        dummy_importance[c] = {}
        for node in best_network.nodes():
            dummy_importance[c][node] = np.random.random()
        # or if you want them all =0 => dummy_importance[c][node] = 0

    # 6) Build side-by-side subplots with your code
    subplots_encoded = clusters_dag_as_base64(best_network, dummy_importance, cluster_names)
    subplots_img_src = "data:image/png;base64," + subplots_encoded

    # 7) Return in the layout
    arcs_list = list(best_network.arcs())
    return html.Div([
        html.H4("Clustering Completed (Network Only)"),
        html.P(f"Number of clusters = {k_clusters}"),
        html.H5("Arcs:"),
        html.Ul([html.Li(str(arc)) for arc in arcs_list]),

        # Single BN figure
        html.Img(src=single_bn_src, style={'maxWidth': '400px'}),
        
        html.Hr(),
        html.H4("Side-by-Side Clusters Dag"),
        html.Img(src=subplots_img_src, style={'maxWidth': '1200px'})
    ])
if __name__ == '__main__':
    print("[DEBUG] Starting Dash server on port 8055...")
    app.run_server(debug=True, host='0.0.0.0', port=8055)