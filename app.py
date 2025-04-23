from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import joblib

# Load the trained XGBClassifier model
model = joblib.load('models/xgb_model.pkl')

# Define the feature columns used in the model
feature_columns = [
    'Age', 'BusinessTravel', 'DailyRate', 'Department', 'DistanceFromHome',
    'Education', 'EducationField', 'EnvironmentSatisfaction', 'Gender',
    'HourlyRate', 'JobInvolvement', 'JobLevel', 'JobRole',
    'JobSatisfaction', 'MaritalStatus', 'MonthlyIncome', 'MonthlyRate',
    'NumCompaniesWorked', 'OverTime', 'PercentSalaryHike',
    'PerformanceRating', 'RelationshipSatisfaction', 'StandardHours',
    'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
    'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole',
    'YearsSinceLastPromotion', 'YearsWithCurrManager'
]

# Initialize the Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout of the app
app.layout = dbc.Container([
    html.H1("Employee Attrition Prediction", className="text-center my-4"),
    dbc.Form([
        dbc.Row([
            dbc.Col([
                dbc.Label("Age"),
                dbc.Input(id="age", type="number", placeholder="Enter age", required=True),
            ]),
            dbc.Col([
                dbc.Label("Business Travel"),
                dbc.Select(
                    id="business_travel",
                    options=[
                        {"label": "Non-Travel", "value": 2},
                        {"label": "Travel Rarely", "value": 0},
                        {"label": "Travel Frequently", "value": 1},
                    ],
                    required=True
                ),
            ]),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Label("Daily Rate"),
                dbc.Input(id="daily_rate", type="number", placeholder="Enter daily rate", required=True),
            ]),
            dbc.Col([
                dbc.Label("Department"),
                dbc.Select(
                    id="department",
                    options=[
                        {"label": "Sales", "value": 0},
                        {"label": "Research & Development", "value": 1},
                        {"label": "Human Resources", "value": 2},
                    ],
                    required=True
                ),
            ]),
        ]),
        # Add more input fields for other features as needed
        dbc.Row([
            dbc.Col([
                dbc.Button("Predict", id="predict-btn", color="primary", className="mt-3"),
            ]),
        ]),
    ]),
    html.Hr(),
    html.Div(id="prediction-output", className="mt-4"),
])

# Define the callback for prediction
@app.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    [State("age", "value"),
     State("business_travel", "value"),
     State("daily_rate", "value"),
     State("department", "value")]
    # Add more states for other features
)
def predict_attrition(n_clicks, age, business_travel, daily_rate, department):
    if n_clicks is None:
        return ""

    # Create a DataFrame with the input values
    input_data = pd.DataFrame([[
        age, business_travel, daily_rate, department
        # Add other feature values here
    ]], columns=feature_columns)

    # Predict the attrition probability
    probability = model.predict_proba(input_data)[0][1]

    return html.Div([
        html.H4(f"Attrition Probability: {probability:.2%}", className="text-success")
    ])

# Expose the server for deployment
server = app.server

# Run the app (only for local development)
if __name__ == "__main__":
    app.run(debug=True)
