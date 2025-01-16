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

# === Import your code ===
import discrete_structure
import discrete_analysis_hellinger
import discrete_representation
import pybnesianCPT_to_df

# For demonstration, let's say 'customers.py' also has some utility
# to build an initial BN or read the dataset. We'll pseudo-import it:
#import customers

# If you want to load the already trained network for reference:
# import pybnesian as pb

# *** NOTE ***: You will need to adapt these imports, functions calls, and paths 
# depending on how your modules are structured and how your user is supposed to call them.

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # For deploying if needed

# ----- Layout -----

app.layout = html.Div([
    html.H1("Bayesian Network Clustering Demo", style={'textAlign': 'center'}),
    html.Hr(),

    # 1) Data Upload
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

    # 2) Select an Action
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

    # 3) Parameters for "Cluster Network Only"
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

    # 4) Parameters for "Cluster Network + Importance Analysis"
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
        # Final run for "Cluster + Importance"
        html.Button("Run Clustering + Importance", id='run-clustering-importance-button', n_clicks=0, style={'display': 'none', 'marginTop': '10px'})
    ], id='cluster-importance-params', style={'textAlign': 'center', 'display': 'none'}),

    html.Hr(),

    # 5) Output area: images, cluster info, CPTs, etc.
    html.Div(id='output-area', style={'textAlign': 'center'}),

    # Hidden storage
    dcc.Store(id='stored-data'),       # For the user dataset
    dcc.Store(id='stored-dataframe'),  # Pandas dataframe once discretized
    dcc.Store(id='variable-list'),     # For variable ordering
    dcc.Store(id='cluster-results')    # Could store results for the slider, etc.
])


# ===== Callbacks =====

# A) Show/Hide param sections
@app.callback(
    [Output('cluster-only-params', 'style'),
     Output('cluster-importance-params', 'style')],
    Input('action-radio', 'value')
)
def toggle_param_sections(action_value):
    if action_value == 'cluster_only':
        return ({'textAlign': 'center', 'display': 'block'}, {'textAlign': 'center', 'display': 'none'})
    else:
        return ({'textAlign': 'center', 'display': 'none'}, {'textAlign': 'center', 'display': 'block'})


# B) Upload Data
@app.callback(
    Output('upload-data-filename', 'children'),
    Output('stored-data', 'data'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename')
)
def handle_upload(content, filename):
    if content is None:
        raise dash.exceptions.PreventUpdate
    # Basic parse of CSV
    content_type, content_string = content.split(',')
    decoded = base64.b64decode(content_string)
    return (f"File uploaded: {filename}", decoded.decode('utf-8'))


# C) Once data is stored, parse into DataFrame
@app.callback(
    Output('stored-dataframe', 'data'),
    Input('stored-data', 'data')
)
def parse_to_dataframe(raw_text):
    if raw_text is None:
        raise dash.exceptions.PreventUpdate
    import io
    # We assume CSV
    df = pd.read_csv(io.StringIO(raw_text))
    # The user says “discrete only” but we might do some checks or forced cat conversion
    for col in df.columns:
        df[col] = df[col].astype('category')
    return df.to_json(date_format='iso', orient='split')


# D) If "Manually Select Variable Order" is chosen -> show dropdowns for each variable
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
    var_list = df.columns.tolist()  # except cluster? The user might define it. Up to you.
    # Build a series of dropdowns for each variable to pick a rank 1..N
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


# E) “Continue” button -> show/hide the “Run Clustering + Importance” button
@app.callback(
    Output('run-clustering-importance-button', 'style'),
    Input('continue-importance-button', 'n_clicks'),
    State('variable-order-radio', 'value'),
    State({'type': 'var-order-dropdown', 'index': ALL}, 'value')
)
def show_final_run_button(n_clicks, order_choice, all_values):
    if n_clicks is None or n_clicks == 0:
        return {'display': 'none'}
    # If user chose manual but not all positions selected, we might handle that
    if order_choice == 'manual':
        if None in all_values or len(set([v for v in all_values if v is not None])) != len(all_values):
            # user has duplicates or missing selections
            return {'display': 'none'}  # block user from continuing
    return {'display': 'inline-block'}


# F) “Run Clustering” (Cluster Only)
@app.callback(
    Output('output-area', 'children'),
    Input('run-clustering-button', 'n_clicks'),
    State('num-clusters-input', 'value'),
    State('stored-dataframe', 'data')
)
def run_cluster_only(n_clicks, k_clusters, df_json):
    if n_clicks is None or n_clicks == 0:
        raise dash.exceptions.PreventUpdate
    if not df_json:
        return "No dataset found. Please upload a CSV."

    # 1) Read DataFrame
    df = pd.read_json(df_json, orient='split')
    # 2) Build naive BN with 'cluster' node and arcs = (cluster->var) for each variable
    #    Then call your structure SEM or partial logic from discrete_structure or from your 'customers.py' example
    #    For brevity, let's just pseudo-code:
    # NOTE: In your 'customers.py' logic, you do something like:
    """
    # in_arcs = [('cluster', var) for var in df.columns]
    # in_nodes = ['cluster'] + list(df.columns)
    # (We also define categories dict)
    # red_inicial = pb.DiscreteBN(in_nodes, in_arcs)
    # clusters = [f'c{i}' for i in range(1, k_clusters+1)]
    # best = sem(...) # from discrete_structure.py
    """
    # We'll mock some results here:
    # but you'd do something akin to the code in customers.py + discrete_structure.sem(...) 
    try:
        # For demonstration, let's say we call a function sem(...) from discrete_structure
        # best_network = discrete_structure.sem(red_inicial, df, categories, clusters)
        # We'll just pretend we got "best_network"
        # Then, we might produce cluster images. For example, we might produce a radar chart or show arcs, etc.
        result_divs = [
            html.H4("Clustering Completed"),
            html.P(f"Number of clusters = {k_clusters}"),
            # Possibly show some images or placeholders
            html.Img(src='data:image/png;base64,XYZ', style={'maxWidth': '300px'})
        ]
        return result_divs
    except Exception as e:
        return f"Error in cluster_only: {str(e)}"


# G) “Run Clustering + Importance”
@app.callback(
    Output('output-area', 'children'),
    Input('run-clustering-importance-button', 'n_clicks'),
    State('num-clusters-importance-input', 'value'),
    State('num-samples-input', 'value'),
    State('variable-order-radio', 'value'),
    State({'type': 'var-order-dropdown', 'index': ALL}, 'value'),
    State({'type': 'var-order-dropdown', 'index': ALL}, 'id'),
    State('stored-dataframe', 'data')
)
def run_cluster_importance(n_clicks, k_clusters, n_samples, order_choice, all_values, all_ids, df_json):
    if n_clicks is None or n_clicks == 0:
        raise dash.exceptions.PreventUpdate
    if not df_json:
        return "No dataset found. Please upload a CSV."

    df = pd.read_json(df_json, orient='split')
    # Possibly reorder columns if user chose manual variable ordering
    if order_choice == 'manual':
        # user provided a rank for each var
        var_positions = {}
        for val, id_ in zip(all_values, all_ids):
            var_positions[id_['index']] = val
        # sort by position
        sorted_vars = sorted(var_positions.items(), key=lambda x: x[1])
        # reorder df columns
        new_col_order = [s[0] for s in sorted_vars]
        # we might skip 'cluster' if it already exists
        # but presumably it doesn't exist yet in the data
        df = df[new_col_order]

    # Next: run a function that does the BN structure + importance measures from:
    #   discrete_analysis_hellinger, discrete_representation, etc.
    try:
        # 1) Learn BN with k_clusters
        # 2) Sample or get MAP reps
        # 3) Compute importance
        # 4) Generate images (radar charts, network arcs, etc.)
        #    We'll do a quick mock for demonstration:
        
        # Example: get MAP representatives
        #   maps = discrete_analysis_hellinger.get_MAP(best_network, clusters, n_samples)
        #   Then naming, naming_categories, or cluster_dag
        # For the actual code, see your customers_analysis.py calls.

        # We'll pretend we generate a figure and store in memory
        fig, ax = plt.subplots()
        ax.set_title("Importance Analysis Example")
        ax.plot(np.random.rand(10))
        # Convert fig to base64
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)
        encoded = base64.b64encode(buf.read()).decode('utf-8')
        img_src = f"data:image/png;base64,{encoded}"

        # Return results
        cluster_info = html.Div([
            html.H4("Cluster + Importance Analysis Results"),
            html.P(f"Number of clusters = {k_clusters}"),
            html.P(f"Samples for inference = {n_samples}"),
            html.Img(src=img_src, style={'maxWidth': '400px'}),
            html.Hr(),
            html.Div("Below: CPTs / cluster representation... (mock data)")
        ])
        return cluster_info

    except Exception as e:
        return f"Error in cluster_importance: {str(e)}"


# If you want to incorporate a “slider” or “arrows” to navigate among cluster images:
# - You could store each cluster's figure in a list, then display based on a slider value or button clicks.
# - This is an advanced topic but entirely doable. You’d store them in a dcc.Store or a global variable,
#   then in a callback read the index from dcc.Slider / or arrow buttons to display the correct image.


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
