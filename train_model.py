"""
A Comprehensive Measure of Well-Being — Model Training Pipeline
Matches the official project Instructions:
  1. Environment Setup & Package Installation
  2. Dataset Collection & Understanding
  3. Data Visualization & Analysis (strip plots, distplots, heatmaps, scatterplots)
  4. Data Preprocessing & Feature Engineering (mean imputation, label encoding, split)
  5. Machine Learning Model Building (Linear Regression, R^2, actual vs predicted)
  6. Model Saving & Serialization (Pickle)
"""

# ---------- 1. Required Libraries ----------
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

os.makedirs("static", exist_ok=True)
os.makedirs("model", exist_ok=True)

# ---------- 2. Dataset Collection & Understanding ----------
df = pd.read_csv("data/hdi_dataset.csv")

print("=== Shape ===", df.shape)
print("\n=== Head ===\n", df.head())
print("\n=== Info ===")
df.info()
print("\n=== Describe ===\n", df.describe())
print("\n=== Nulls before cleaning ===\n", df.isnull().sum())

# ---------- 3. Data Visualization & Analysis ----------
numeric_cols = ["life_expectancy", "mean_years_schooling",
                 "expected_years_schooling", "gni_per_capita", "hdi_score"]

# Strip plot
plt.figure(figsize=(8, 5))
sns.stripplot(data=df[numeric_cols])
plt.title("Strip Plot of Numeric Features")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig("static/strip_plot.png")
plt.close()

# Distribution plots
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, col in zip(axes.flat, ["life_expectancy", "mean_years_schooling",
                                "gni_per_capita", "hdi_score"]):
    sns.histplot(df[col].dropna(), kde=True, ax=ax, color="#2563eb")
    ax.set_title(f"Distribution: {col}")
plt.tight_layout()
plt.savefig("static/dist_plots.png")
plt.close()

# Heatmap / correlation matrix
plt.figure(figsize=(7, 6))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("static/correlation_heatmap.png")
plt.close()

# Scatter plots (each feature vs hdi_score)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["life_expectancy", "mean_years_schooling", "gni_per_capita"]):
    sns.scatterplot(data=df, x=col, y="hdi_score", ax=ax, color="#16a34a")
    ax.set_title(f"{col} vs HDI Score")
plt.tight_layout()
plt.savefig("static/scatter_plots.png")
plt.close()

print("\nSaved EDA visualizations to static/ folder.")

# ---------- 4. Data Preprocessing & Feature Engineering ----------
# Mean imputation for missing numeric values
for col in ["life_expectancy", "mean_years_schooling", "gni_per_capita"]:
    df[col] = df[col].fillna(df[col].mean())

# hdi_score / hdi_category may still be null only if source row was fully broken;
# recompute is unnecessary here since only feature columns had NaNs injected.
df["hdi_score"] = df["hdi_score"].fillna(df["hdi_score"].mean())

print("\n=== Nulls after mean imputation ===\n", df.isnull().sum())

# Label Encoding (country_name is categorical, kept for reference/dropdown -
# encoded version demonstrates the required label-encoding step)
le = LabelEncoder()
df["country_encoded"] = le.fit_transform(df["country_name"])

# Selecting Dependent (target) and Independent (features) Variables
features = ["life_expectancy", "mean_years_schooling", "expected_years_schooling", "gni_per_capita"]
target = "hdi_score"

X = df[features]
y = df[target]

# Train/Test Split (75/25 as specified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print(f"\nTrain size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")

# ---------- 5. Machine Learning Model Building ----------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)

print(f"\n=== Model Evaluation ===")
print(f"Train R^2: {train_r2:.4f}")
print(f"Test R^2:  {test_r2:.4f}")
print(f"Test MSE:  {mse:.5f}")
print(f"Coefficients: {dict(zip(features, model.coef_))}")
print(f"Intercept: {model.intercept_:.4f}")

# Actual vs Predicted scatter plot
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred_test, color="#2563eb", alpha=0.7)
plt.plot([0, 1], [0, 1], color="red", linestyle="--")
plt.xlabel("Actual HDI Score")
plt.ylabel("Predicted HDI Score")
plt.title(f"Actual vs Predicted (R² = {test_r2:.3f})")
plt.tight_layout()
plt.savefig("static/actual_vs_predicted.png")
plt.close()

# ---------- 6. Model Saving & Serialization (Pickle) ----------
with open("model/hdi_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("\nModel saved to model/hdi_model.pkl (pickle serialization)")


def categorize(score: float) -> str:
    if score >= 0.80:
        return "Very High"
    elif score >= 0.70:
        return "High"
    elif score >= 0.55:
        return "Medium"
    else:
        return "Low"


sample = pd.DataFrame({
    "actual_score": y_test.values[:5],
    "predicted_score": y_pred_test[:5],
})
sample["predicted_category"] = sample["predicted_score"].apply(categorize)
print("\n=== Sample Predictions ===\n", sample)

# Save cleaned dataset (with country dropdown data) for the Flask app to use
df[["country_name"] + features].to_csv("data/countries_for_dropdown.csv", index=False)
