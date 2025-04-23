from dash import Dash, dcc, html
import plotly.express as px
import pandas as pd

# Initialize the Dash app
app = Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("My Dash App"),
    dcc.Graph(
        id='example-graph',
        figure=px.scatter(
            pd.DataFrame({
                "x": [1, 2, 3, 4],
                "y": [10, 11, 12, 13],
                "label": ["A", "B", "C", "D"]
            }),
            x="x", y="y", color="label", title="Example Scatter Plot"
        )
    )
])

# Expose the server for gunicorn
server = app.server

# Run the app (only for local development)
if __name__ == "__main__":
    app.run_server(debug=True)

