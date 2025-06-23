from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import random

app = Flask(__name__)

df = pd.read_csv("player_performance_updated.csv")
df_encoded = pd.get_dummies(df[["Player", "Opposition", "Venue"]])

@app.route('/')
def index():
    roles = sorted(df["Role"].unique())
    venues = sorted(df["Venue"].unique())
    oppositions = sorted(df["Opposition"].unique())
    return render_template("index.html", roles=roles, venues=venues, oppositions=oppositions)

@app.route("/predict", methods=["POST"])
def predict():
    role = request.form.get("role")
    player = request.form.get("player")
    venue = request.form.get("venue")
    opposition = request.form.get("opposition")

    filtered_df = df[df["Role"] == role]
    player_data = filtered_df[filtered_df["Player"] == player]
    recent_form_list = [int(x.strip()) for x in player_data["Recent_Form"].values[0].strip("[]").split(',')]
    avg_form = player_data["Avg_Recent_Form"].values[0]

    X = df_encoded
    if role == "Batsman":
        y = df["Predicted_Runs"]
    elif role == "Bowler":
        y = df["Predicted_Wickets"]
    else:
        y_runs = df["Predicted_Runs"]
        y_wkts = df["Predicted_Wickets"]

    if role != "All-Rounder":
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

    else:
        X_train, _, y_runs_train, _ = train_test_split(X, y_runs, test_size=0.2, random_state=42)
        _, _, y_wkts_train, _ = train_test_split(X, y_wkts, test_size=0.2, random_state=42)

        model_runs = RandomForestRegressor(n_estimators=100, random_state=42)
        model_wkts = RandomForestRegressor(n_estimators=100, random_state=42)

        model_runs.fit(X_train, y_runs_train)
        model_wkts.fit(X_train, y_wkts_train)


    input_df = pd.DataFrame([[player, opposition, venue]], columns=["Player", "Opposition", "Venue"])
    input_encoded = pd.get_dummies(input_df)

    for col in X.columns:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    input_encoded = input_encoded[X.columns]

    prediction = None
    prediction_wickets = None
    if role == "Batsman":
        prediction = model.predict(input_encoded)[0]
    elif role == "Bowler":
        prediction_wickets = int(round(model.predict(input_encoded)[0]))
    else:
        prediction = model_runs.predict(input_encoded)[0]
        prediction_wickets = int(round(model_wkts.predict(input_encoded)[0]))
    return render_template("index.html",
                           roles=sorted(df["Role"].unique()),
                           venues=sorted(df["Venue"].unique()),
                           oppositions=sorted(df["Opposition"].unique()),
                           selected_role=role,
                           selected_player=player,
                           selected_venue=venue,
                           selected_opposition=opposition,
                           recent_form_list=recent_form_list,
                           avg_form=avg_form,
                           prediction=prediction,
                           prediction_wickets=prediction_wickets)

@app.route("/get_players", methods=["POST"])
def get_players():
    role = request.form.get("role")
    filtered_df = df[df["Role"] == role]
    players = sorted(filtered_df["Player"].unique())
    return {"players": players}

if __name__ == '__main__':
    app.run(debug=True)
