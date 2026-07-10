"""
Building the Flask Web Application
- Home Page Introduction
- HDI Prediction Interface with Country Selection Dropdown + manual input
- User Input Forms
- Prediction Result Display (separate result page)
"""
from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

with open("model/hdi_model.pkl", "rb") as f:
    model = pickle.load(f)

countries_df = pd.read_csv("data/countries_for_dropdown.csv")


def categorize(score: float) -> str:
    if score >= 0.80:
        return "Very High"
    elif score >= 0.70:
        return "High"
    elif score >= 0.55:
        return "Medium"
    else:
        return "Low"


@app.route("/", methods=["GET"])
def index():
    countries = countries_df["country_name"].tolist()
    return render_template("indexnew.html", countries=countries)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        selected_country = request.form.get("country_select")

        if selected_country and selected_country != "manual":
            row = countries_df[countries_df["country_name"] == selected_country].iloc[0]
            life_expectancy = float(row["life_expectancy"])
            mean_years_schooling = float(row["mean_years_schooling"])
            expected_years_schooling = float(row["expected_years_schooling"])
            gni_per_capita = float(row["gni_per_capita"])
        else:
            selected_country = "Custom Input"
            life_expectancy = float(request.form["life_expectancy"])
            mean_years_schooling = float(request.form["mean_years_schooling"])
            expected_years_schooling = float(request.form["expected_years_schooling"])
            gni_per_capita = float(request.form["gni_per_capita"])

        features = pd.DataFrame([{
            "life_expectancy": life_expectancy,
            "mean_years_schooling": mean_years_schooling,
            "expected_years_schooling": expected_years_schooling,
            "gni_per_capita": gni_per_capita,
        }])

        predicted_score = float(np.clip(model.predict(features)[0], 0, 1))
        category = categorize(predicted_score)

        result = {
            "country": selected_country,
            "life_expectancy": life_expectancy,
            "mean_years_schooling": mean_years_schooling,
            "expected_years_schooling": expected_years_schooling,
            "gni_per_capita": gni_per_capita,
            "score": round(predicted_score, 3),
            "category": category,
            "error": None
        }
    except (ValueError, KeyError, IndexError):
        result = {"error": "Please provide valid inputs or select a country."}

    return render_template("resultnew.html", result=result)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
