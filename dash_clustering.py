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
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)


############################################################
# HELPER: Make a DAG figure from a learned BN using networkx
############################################################
def plot_bn_dag(bn: pb.DiscreteBN, title="Learned BN Structure"):
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
            # If get_MAP returns (category, prob)
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
    return f"data:image/png;base64,{encoded}"


# ====================== LAYOUT ======================
app.layout = dcc.Loading(
    id="global-spinner",
    overlay_style={"visibility":"visible", "filter": "blur(1px)"},
    type="circle",        # You can choose "circle", "dot", "default", etc.
    fullscreen=False,      # This ensures it covers the entire page
    children=html.Div([
    html.H1("Bayesian Network Clustering", style={'textAlign': 'center'}),
    # Upload
    html.Div([
        html.H3("Upload Discrete Dataset (CSV)"),
        dcc.Upload(
            id='upload-data',
            children=html.Button('Upload File (CSV)'),
            multiple=False
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
        html.Button("Run Clustering + Importance", id='run-clustering-importance-button', n_clicks=0,
                    style={'display': 'none', 'marginTop': '10px'})
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
    if action_value == 'cluster_only':
        return ({'textAlign': 'center', 'display': 'block'},
                {'textAlign': 'center', 'display': 'none'})
    else:
        return ({'textAlign': 'center', 'display': 'none'},
                {'textAlign': 'center', 'display': 'block'})


# ============ Upload Parsing ============

@app.callback(
    Output('upload-data-filename', 'children'),
    Output('stored-data', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_upload(content, filename):
    if content is None:
        raise dash.exceptions.PreventUpdate

    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    msg = f"File uploaded: {filename}"
    return msg, decoded.decode('utf-8')


@app.callback(
    Output('stored-dataframe', 'data'),
    Input('stored-data', 'data')
)
def parse_to_dataframe(raw_text):
    if raw_text is None:
        raise dash.exceptions.PreventUpdate

    import io
    # Read entire CSV as strings
    df = pd.read_csv(io.StringIO(raw_text), dtype=str)
    
    # Strip column names in case of trailing spaces
    df.columns = [c.strip() for c in df.columns]

    # Convert EVERY column to category
    # Even "ID" or numeric columns remain, but become many-categories
    for col in df.columns:
        # Force to string (again) then to category
        df[col] = df[col].astype(str).astype('category')

    print("DEBUG: Final columns ->", list(df.columns))
    print("DEBUG: dtypes ->\n", df.dtypes)

    # Return as JSON
    return df.to_json(date_format='iso', orient='split')



# ============ Manual order ============

@app.callback(
    [Output('manual-order-container', 'children'),
     Output('manual-order-container', 'style')],
    Input('variable-order-radio', 'value'),
    State('stored-dataframe', 'data')
)
def show_manual_order(order_choice, df_json):
    if not df_json or order_choice != 'manual':
        return ([], {'display': 'none'})

    df = pd.read_json(df_json, orient='split')
    var_list = df.columns.tolist()

    children = []
    used_positions = list(range(1, len(var_list)+1))
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
    if n_clicks is None or n_clicks == 0:
        return {'display': 'none'}

    if order_choice == 'manual':
        if None in all_values or len(set([v for v in all_values if v is not None])) != len(all_values):
            return {'display': 'none'}

    return {'display': 'inline-block'}


# ============ Cluster-Only ============

@app.callback(
    Output('output-area', 'children',
           allow_duplicate=True),  # We assume Dash>=2.9
    Input('run-clustering-button', 'n_clicks'),
    State('num-clusters-input', 'value'),
    State('stored-dataframe', 'data'),
    prevent_initial_call='initial_duplicate'
)
def run_cluster_only(n_clicks, k_clusters, df_json):
    if n_clicks is None or n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    if not df_json:
        return "No dataset found. Please upload a CSV."

    df = pd.read_json(df_json, orient='split')

    for col in df.columns:
        df[col] = df[col].astype(str).astype('category')


    # Build naive BN with cluster->var arcs
    in_arcs = [('cluster', var) for var in df.columns]
    in_nodes = ['cluster'] + list(df.columns)
    cluster_names = [f'c{i}' for i in range(1, k_clusters+1)]

    red_inicial = pb.DiscreteBN(in_nodes, in_arcs)

    # Build categories
    categories = {'cluster': cluster_names}
    for var in df.columns:
        print(f"DEBUG: Accessing df['{var}'] => dtype={df[var].dtype}")
        categories[var] = df[var].cat.categories.tolist()

    best_network = discrete_structure.sem(red_inicial, df, categories, cluster_names)
    arcs_list = list(best_network.arcs())

    dag_img_src = plot_bn_dag(best_network, "Learned BN (Cluster Only)")
    result_divs = [
        html.H4("Clustering Completed (Network Only)"),
        html.P(f"Number of clusters = {k_clusters}"),
        html.H5("Arcs:"),
        html.Ul([html.Li(str(arc)) for arc in arcs_list]),
        html.Img(src=dag_img_src, style={'maxWidth': '500px'})
    ]
    return result_divs


# ============ Cluster + Importance ============

@app.callback(
    Output('output-area', 'children',
           allow_duplicate=True),
    Input('run-clustering-importance-button', 'n_clicks'),
    State('num-clusters-importance-input', 'value'),
    State('num-samples-input', 'value'),
    State('variable-order-radio', 'value'),
    State({'type': 'var-order-dropdown', 'index': ALL}, 'value'),
    State({'type': 'var-order-dropdown', 'index': ALL}, 'id'),
    State('stored-dataframe', 'data'),
    prevent_initial_call='initial_duplicate'
)
def run_cluster_importance(n_clicks, k_clusters, n_samples, order_choice, all_values, all_ids, df_json):
    if n_clicks is None or n_clicks == 0:
        raise dash.exceptions.PreventUpdate

    if not df_json:
        return "No dataset found. Please upload a CSV."

    df = pd.read_json(df_json, orient='split')
    
    for col in df.columns:
        df[col] = df[col].astype(str).astype('category')

    # Reorder columns if manual
    if order_choice == 'manual':
        var_positions = {}
        for val, id_ in zip(all_values, all_ids):
            var_positions[id_['index']] = val
        sorted_vars = sorted(var_positions.items(), key=lambda x: x[1])
        new_col_order = [s[0] for s in sorted_vars]
        df = df[new_col_order]

    in_arcs = [('cluster', var) for var in df.columns]
    in_nodes = ['cluster'] + list(df.columns)
    cluster_names = [f'c{i}' for i in range(1, k_clusters+1)]

    red_inicial = pb.DiscreteBN(in_nodes, in_arcs)

    categories = {'cluster': cluster_names}
    for var in df.columns:
        categories[var] = df[var].cat.categories.tolist()

    best_network = discrete_structure.sem(red_inicial, df, categories, cluster_names)

    # get MAP
    map_reps = discrete_analysis_hellinger.get_MAP(best_network, cluster_names, n=n_samples)

    # compute importance
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

    # Build df_categories
    df_categories = discrete_analysis_hellinger.df_to_dict(df)

    radar_img_src = plot_map_with_importance(map_reps, importances_dict, df_categories)
    dag_img_src = plot_bn_dag(best_network, "Cluster + Importance BN")

    arcs_list = list(best_network.arcs())

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
    return layout_div


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
