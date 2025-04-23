from dash import Dash, dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import joblib

# Load the trained XGBClassifier model
model = joblib.load('models/xgb_model.pkl')

# Define the feature columns used in the model (match training data)
feature_columns = [
    'Age', 'Education', 'EnvironmentSatisfaction', 'JobInvolvement',
    'JobLevel', 'JobSatisfaction', 'MonthlyIncome', 'OverTime',
    'RelationshipSatisfaction', 'StandardHours', 'StockOptionLevel',
    'TotalWorkingYears', 'TrainingTimesLastYear', 'WorkLifeBalance',
    'YearsAtCompany', 'YearsInCurrentRole', 'YearsWithCurrManager'
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
                dbc.Label("Education"),
                dbc.Input(id="education", type="number", placeholder="Enter education level", required=True),
            ]),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Label("Environment Satisfaction"),
                dbc.Input(id="environment_satisfaction", type="number", placeholder="Enter environment satisfaction", required=True),
            ]),
            dbc.Col([
                dbc.Label("Job Involvement"),
                dbc.Input(id="job_involvement", type="number", placeholder="Enter job involvement", required=True),
            ]),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Label("Job Level"),
                dbc.Input(id="job_level", type="number", placeholder="Enter job level", required=True),
            ]),
            dbc.Col([
                dbc.Label("Job Satisfaction"),
                dbc.Input(id="job_satisfaction", type="number", placeholder="Enter job satisfaction", required=True),
            ]),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Label("Monthly Income"),
                dbc.Input(id="monthly_income", type="number", placeholder="Enter monthly income", required=True),
            ]),
            dbc.Col([
                dbc.Label("Over Time"),
                dbc.Select(
                    id="over_time",
                    options=[
                        {"label": "Yes", "value": 1},
                        {"label": "No", "value": 0},
                    ],
                    required=True
                ),
            ]),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Label("Relationship Satisfaction"),
                dbc.Input(id="relationship_satisfaction", type="number", placeholder="Enter relationship satisfaction", required=True),
            ]),
            dbc.Col([
                dbc.Label("Standard Hours"),
                dbc.Input(id="standard_hours", type="number", placeholder="Enter standard hours", required=True),
            ]),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Label("Stock Option Level"),
                dbc.Input(id="stock_option_level", type="number", placeholder="Enter stock option level", required=True),
            ]),
            dbc.Col([
                dbc.Label("Total Working Years"),
                dbc.Input(id="total_working_years", type="number", placeholder="Enter total working years", required=True),
            ]),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Label("Training Times Last Year"),
                dbc.Input(id="training_times_last_year", type="number", placeholder="Enter training times last year", required=True),
            ]),
            dbc.Col([
                dbc.Label("Work Life Balance"),
                dbc.Input(id="work_life_balance", type="number", placeholder="Enter work life balance", required=True),
            ]),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Label("Years At Company"),
                dbc.Input(id="years_at_company", type="number", placeholder="Enter years at company", required=True),
            ]),
            dbc.Col([
                dbc.Label("Years In Current Role"),
                dbc.Input(id="years_in_current_role", type="number", placeholder="Enter years in current role", required=True),
            ]),
        ]),
        dbc.Row([
            dbc.Col([
                dbc.Label("Years With Current Manager"),
                dbc.Input(id="years_with_curr_manager", type="number", placeholder="Enter years with current manager", required=True),
            ]),
        ]),
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
     State("education", "value"),
     State("environment_satisfaction", "value"),
     State("job_involvement", "value"),
     State("job_level", "value"),
     State("job_satisfaction", "value"),
     State("monthly_income", "value"),
     State("over_time", "value"),
     State("relationship_satisfaction", "value"),
     State("standard_hours", "value"),
     State("stock_option_level", "value"),
     State("total_working_years", "value"),
     State("training_times_last_year", "value"),
     State("work_life_balance", "value"),
     State("years_at_company", "value"),
     State("years_in_current_role", "value"),
     State("years_with_curr_manager", "value")]
)
def predict_attrition(n_clicks, age, education, environment_satisfaction, job_involvement,
                      job_level, job_satisfaction, monthly_income, over_time,
                      relationship_satisfaction, standard_hours, stock_option_level,
                      total_working_years, training_times_last_year, work_life_balance,
                      years_at_company, years_in_current_role, years_with_curr_manager):
    if n_clicks is None:
        return ""

    # Check for missing values
    if None in [age, education, environment_satisfaction, job_involvement, job_level,
                job_satisfaction, monthly_income, over_time, relationship_satisfaction,
                standard_hours, stock_option_level, total_working_years,
                training_times_last_year, work_life_balance, years_at_company,
                years_in_current_role, years_with_curr_manager]:
        return html.Div("Please fill in all the required fields.", className="text-danger")

    # Convert OverTime to integer
    try:
        over_time = int(over_time)
    except ValueError:
        return html.Div("Invalid value for OverTime.", className="text-danger")

    # Create a DataFrame with the input values
    try:
        input_data = pd.DataFrame([
            [age, education, environment_satisfaction, job_involvement, job_level,
             job_satisfaction, monthly_income, over_time, relationship_satisfaction,
             standard_hours, stock_option_level, total_working_years,
             training_times_last_year, work_life_balance, years_at_company,
             years_in_current_role, years_with_curr_manager]
        ], columns=feature_columns)
    except Exception as e:
        return html.Div(f"Error creating input data: {str(e)}", className="text-danger")

    # Predict the attrition probability
    try:
        probability = model.predict_proba(input_data)[0][1]
    except Exception as e:
        return html.Div(f"Error during prediction: {str(e)}", className="text-danger")

    return html.Div([
        html.H4(f"Attrition Probability: {probability:.2%}", className="text-success")
    ])

# Expose the server for deployment
server = app.server

# Run the app (only for local development)
if __name__ == "__main__":
    app.run(debug=True)
