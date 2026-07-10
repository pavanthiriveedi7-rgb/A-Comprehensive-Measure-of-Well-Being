# A Comprehensive Measure of Well-Being — HDI Prediction Platform

A machine learning web app that predicts a country's **Human Development Index (HDI)**
score and classifies it into **Very High / High / Medium / Low**, using Linear
Regression on life expectancy, education, and income indicators. Built with
Python, Flask, and scikit-learn.

---

## 1. Technical Architecture

```
User Layer (Researcher / Policy Maker / Student / Government)
        │
        ▼
Frontend Layer (indexnew.html — country dropdown + manual input form)
        │  POST /predict
        ▼
Flask Application Layer (app.py — routing, validation, prediction)
        │
        ▼
Model Storage Layer (model/hdi_model.pkl — Pickle serialized)
        │
        ▼
Prediction Engine → HDI score → category (Very High/High/Medium/Low)
        │
        ▼
Result Page (resultnew.html)
```

**ML Lifecycle:** HDI Dataset (CSV) → EDA (strip plots, distplots, heatmaps,
scatterplots) → Preprocessing (mean imputation, label encoding) → Train/Test
Split (75/25) → Linear Regression → Evaluation (R², actual vs predicted) →
Pickle serialization.

---

## 2. Pre-requisites

**Hardware:** Intel Core i3+, 4GB+ RAM, 10GB free storage, internet connection
**Software:** Python 3.x, pip (Anaconda/Jupyter optional), Flask

```
pandas, numpy, scikit-learn, matplotlib, seaborn, flask
```

---

## 3. Folder Structure

```
hdi_project/
├── data/
│   ├── hdi_dataset.csv
│   └── countries_for_dropdown.csv
├── model/
│   └── hdi_model.pkl
├── static/
│   ├── strip_plot.png
│   ├── dist_plots.png
│   ├── correlation_heatmap.png
│   ├── scatter_plots.png
│   └── actual_vs_predicted.png
├── templates/
│   ├── indexnew.html
│   └── resultnew.html
├── generate_dataset.py
├── train_model.py
├── app.py
├── requirements.txt
└── README.md
```

---

## 4. Project Workflow

1. **Environment Setup & Package Installation** — install libraries (`requirements.txt`)
2. **Dataset Collection & Understanding** — `generate_dataset.py` creates a
   50-country dataset with Country Name, Life Expectancy, Mean/Expected Years
   of Schooling, GNI per Capita, and HDI Score
3. **Data Visualization & Analysis** — strip plots, distribution plots,
   correlation heatmap, scatter plots (`train_model.py`)
4. **Data Preprocessing & Feature Engineering** — mean imputation for nulls,
   label encoding for country name, feature/target selection, 75/25 train-test split
5. **Machine Learning Model Building** — Linear Regression, R² evaluation,
   actual-vs-predicted scatter plot
6. **Model Saving & Serialization** — model saved via **Pickle** to `model/hdi_model.pkl`
7. **Building the Flask Web Application** — home page with country dropdown +
   manual input (`indexnew.html`), prediction handling (`app.py`), result
   display page (`resultnew.html`)

---

## 5. How to Run

```bash
pip install -r requirements.txt
python generate_dataset.py    # creates data/hdi_dataset.csv
python train_model.py         # EDA, preprocessing, trains model, saves model/hdi_model.pkl
python app.py                 # starts Flask server at http://localhost:5000
```

Open `http://localhost:5000`, pick a country from the dropdown (or choose
"Enter custom values") and submit to see the predicted HDI score and category.

---

## 6. Model Performance

- **Train R²:** ~0.89
- **Test R²:** ~0.83
- **Algorithm:** Linear Regression (scikit-learn)
- **Categories:** Very High (≥0.80), High (≥0.70), Medium (≥0.55), Low (<0.55)

---

## 7. Conclusion

This project demonstrates a complete, end-to-end data science workflow —
from raw country-level indicators to a deployed prediction interface. Linear
Regression, while simple, captures the strong relationship between life
expectancy, education, income, and overall human development, achieving solid
R² performance. The Flask app makes this insight accessible to non-technical
users like policymakers and researchers through a simple country-selection
or custom-input interface.

**Possible extensions:** use the full real-world UNDP HDI dataset (all ~190
countries, multiple years), try non-linear models (Random Forest, XGBoost)
for improved accuracy, and deploy the app publicly via Render or Heroku.
