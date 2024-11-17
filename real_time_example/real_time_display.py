import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
from RaceStrategyEngine.utility import driver_styles_for_plotting
import zmq
import threading

app = dash.Dash(__name__)

# Define driver styles
driver_styles = driver_styles_for_plotting
# Layout of the app
app.layout = html.Div([
    html.H1("Race Trace", style={'color': 'white', 'textAlign': 'center'}),
    dcc.Graph(id='live-graph', animate=True, style={'height': '85vh'}),
    dcc.Interval(
        id='interval-component',
        interval=1000,  # in milliseconds
        n_intervals=0
    )
], style={'backgroundColor': 'black', 'height': '100vh'})

# ZeroMQ setup
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect("tcp://localhost:5555")
socket.setsockopt_string(zmq.SUBSCRIBE, "")

# Global variable to store the latest data
latest_data = {}


def zmq_listener():
    global latest_data
    while True:
        latest_data = socket.recv_json()


# Start ZeroMQ listener in a separate thread
threading.Thread(target=zmq_listener, daemon=True).start()


@app.callback(Output('live-graph', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph(n):
    global latest_data

    if not latest_data:
        return {'data': [], 'layout': go.Layout(title='Waiting for data...')}

    current_lap = latest_data.pop('current_lap', 0)

    traces = []
    for driver, gaps in latest_data.items():
        if driver in driver_styles:
            style = driver_styles[driver]
            trace = go.Scatter(
                x=list(range(1, len(gaps) + 1)),
                y=gaps,
                mode='lines+markers',
                name=driver,
                line=dict(color=style['color'], dash=style['line']['dash']),
                marker=dict(symbol=style['marker'], size=8),
                hoverinfo='text',
                text=[f"{driver}: {gap:.2f}s" for gap in gaps]
            )
        else:
            trace = go.Scatter(
                x=list(range(1, len(gaps) + 1)),
                y=gaps,
                mode='lines+markers',
                name=driver
            )
        traces.append(trace)

    # noinspection PyTypeChecker
    layout = go.Layout(
        # title='Race Trace Prediction',
        xaxis_title='Lap number',
        yaxis_title='Time Difference (s)',
        legend_title='Drivers',
        plot_bgcolor='black',
        paper_bgcolor='black',
        font=dict(color='white'),
        xaxis=dict(gridcolor='white', zerolinecolor='white'),
        yaxis=dict(gridcolor='white', zerolinecolor='white', autorange='reversed'),
        hovermode='closest',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        shapes=[
            dict(
                type="line",
                xref="x", yref="paper",
                x0=current_lap, x1=current_lap,
                y0=0, y1=1,
                line=dict(
                    color="#FFD700",
                    width=4,
                    dash="dash",
                )
            )
        ]
    )

    return {'data': traces, 'layout': layout}


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
