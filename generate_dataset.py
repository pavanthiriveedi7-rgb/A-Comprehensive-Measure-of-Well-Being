"""
Generates an HDI dataset with Country Name, Life Expectancy, Mean Years of
Schooling, Expected Years of Schooling, GNI per Capita, and HDI Score —
matching the structure of real-world HDI datasets (e.g. UNDP / Kaggle).
"""
import numpy as np
import pandas as pd

np.random.seed(42)

countries = [
    "Norway", "Switzerland", "Ireland", "Germany", "Australia", "Iceland",
    "Sweden", "Netherlands", "Denmark", "Finland", "Singapore", "United Kingdom",
    "Japan", "South Korea", "France", "Canada", "United States", "Spain",
    "Italy", "Chile", "Argentina", "Brazil", "China", "Turkey", "Mexico",
    "Thailand", "Malaysia", "Sri Lanka", "Vietnam", "Indonesia", "Philippines",
    "India", "Bangladesh", "Egypt", "Kenya", "Nigeria", "Ghana", "Nepal",
    "Pakistan", "Ethiopia", "Uganda", "Haiti", "Yemen", "Chad", "Niger",
    "Mali", "Burundi", "South Sudan", "Afghanistan", "Mozambique"
]

n = len(countries)

# Feature generation loosely correlated with country "development tier" index
tier = np.linspace(1, 0, n) + np.random.normal(0, 0.05, n)  # 1 = most developed
tier = tier.clip(0, 1)

life_expectancy = (50 + tier * 35 + np.random.normal(0, 2, n)).clip(45, 85)
mean_years_schooling = (2 + tier * 11 + np.random.normal(0, 0.8, n)).clip(1, 14)
expected_years_schooling = (5 + tier * 12 + np.random.normal(0, 1, n)).clip(4, 20)
gni_per_capita = (600 + (tier ** 2) * 70000 + np.random.normal(0, 1500, n)).clip(400, 90000)

df = pd.DataFrame({
    "country_name": countries,
    "life_expectancy": life_expectancy.round(1),
    "mean_years_schooling": mean_years_schooling.round(1),
    "expected_years_schooling": expected_years_schooling.round(1),
    "gni_per_capita": gni_per_capita.round(0),
})

# Introduce a handful of missing values (to make null-handling meaningful)
rng = np.random.default_rng(7)
nan_rows = rng.choice(n, size=6, replace=False)
df.loc[nan_rows[:2], "mean_years_schooling"] = np.nan
df.loc[nan_rows[2:4], "gni_per_capita"] = np.nan
df.loc[nan_rows[4:6], "life_expectancy"] = np.nan

# Compute HDI score using simplified UNDP-style sub-index formula
le_idx = (df["life_expectancy"] - 20) / (85 - 20)
edu_idx = ((df["mean_years_schooling"] / 15) + (df["expected_years_schooling"] / 18)) / 2
gni_idx = (np.log(df["gni_per_capita"].clip(lower=1)) - np.log(100)) / (np.log(75000) - np.log(100))

df["hdi_score"] = ((le_idx + edu_idx + gni_idx) / 3).clip(0, 1).round(3)


def categorize(score):
    if pd.isna(score):
        return np.nan
    if score >= 0.80:
        return "Very High"
    elif score >= 0.70:
        return "High"
    elif score >= 0.55:
        return "Medium"
    else:
        return "Low"


df["hdi_category"] = df["hdi_score"].apply(categorize)

df.to_csv("data/hdi_dataset.csv", index=False)
print("Dataset created:", df.shape)
print(df.head(10))
print("\nCategory distribution:")
print(df["hdi_category"].value_counts())
