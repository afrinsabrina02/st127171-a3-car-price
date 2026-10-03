"""
Car price class predictor - Dash web app for A3.
Loads the model exported by the notebook (app/model/car_model.pkl) and predicts one of 4 price classes.
"""
import os

import joblib
import numpy as np
import pandas as pd
from dash import Dash, dcc, html, Input, Output, State

# ------------------------------------------------------------------ load the exported model
BUNDLE = joblib.load(os.path.join(os.path.dirname(__file__), "model", "car_model.pkl"))
W = BUNDLE["W"]                              # weights, shape (n_features + 1, 4), row 0 = bias
PREPROCESSOR = BUNDLE["preprocessor"]        # fitted scaler + one-hot encoder from the notebook
EDGES = BUNDLE["edges"]                      # price bucket edges (5 numbers)
NUMERIC = BUNDLE["numeric_features"]
CATEGORICAL = BUNDLE["categorical_features"]
DEFAULTS = BUNDLE["defaults"]                # training median / most frequent value of every column
CATEGORIES = dict(zip(CATEGORICAL, PREPROCESSOR.named_transformers_["cat"].categories_))


def class_description(c):
    """Human readable price range of class c, from the quartile edges."""
    lo, hi = EDGES[c], EDGES[c + 1]
    if c == 0:
        return f"up to {hi:,.0f}"
    if c == len(EDGES) - 2:
        return f"above {lo:,.0f}"
    return f"{lo:,.0f} to {hi:,.0f}"


def predict_class(values):
    """values: dict column -> value (None allowed, the training default is used instead).
    Returns (class, probabilities)."""
    row = {col: (DEFAULTS[col] if values.get(col) is None else values[col]) for col in NUMERIC + CATEGORICAL}
    X = PREPROCESSOR.transform(pd.DataFrame([row]))      # same preprocessing as in training
    X = np.column_stack([np.ones(len(X)), X])            # intercept column
    z = X @ W
    z = z - z.max(axis=1, keepdims=True)                 # stable softmax
    p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
    return int(np.argmax(p, axis=1)[0]), p[0]


# ------------------------------------------------------------------ page layout
def number_input(col, label):
    return html.Div([html.Label(label), dcc.Input(id=col, type="number", placeholder=f"default {DEFAULTS[col]:g}",
                                                  style={"width": "100%"})], style={"marginBottom": "10px"})


def dropdown(col, label):
    return html.Div([html.Label(label), dcc.Dropdown(id=col, options=[str(c) for c in CATEGORIES[col]],
                                                     value=str(DEFAULTS[col]), clearable=False)],
                    style={"marginBottom": "10px"})


app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H2("Car price class predictor"),
    html.P("Fill in the car details (empty number fields use the typical value of the training data). "
           "The model predicts which of 4 price classes the car belongs to."),
    number_input("year", "Year"),
    number_input("km_driven", "Kilometres driven"),
    number_input("mileage", "Mileage (kmpl)"),
    number_input("engine", "Engine (CC)"),
    number_input("max_power", "Max power (bhp)"),
    number_input("seats", "Seats"),
    html.Div([html.Label("Owner (1 = first owner ... 4 = fourth and above)"),
              dcc.Dropdown(id="owner", options=[1, 2, 3, 4], value=1, clearable=False)],
             style={"marginBottom": "10px"}),
    dropdown("brand", "Brand"),
    dropdown("fuel", "Fuel"),
    dropdown("seller_type", "Seller type"),
    dropdown("transmission", "Transmission"),
    html.Button("Predict", id="predict", n_clicks=0),
    html.H3(id="result", style={"marginTop": "20px"}),
    html.Div(id="details"),
], style={"maxWidth": "480px", "margin": "30px auto", "fontFamily": "sans-serif"})


@app.callback(
    Output("result", "children"), Output("details", "children"),
    Input("predict", "n_clicks"),
    [State(c, "value") for c in NUMERIC + CATEGORICAL],
)
def on_predict(n_clicks, *vals):
    if not n_clicks:
        return "", ""
    cls, probs = predict_class(dict(zip(NUMERIC + CATEGORICAL, vals)))
    rows = [html.Li(f"Class {c} ({class_description(c)}): {probs[c]:.1%}") for c in range(len(probs))]
    return f"Predicted price class: {cls} ({class_description(cls)})", html.Ul(rows)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8050, debug=False)

