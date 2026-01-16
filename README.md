# Retail Demand Forecasting using Apache Spark

## 📌 Overview
This project implements an end-to-end retail demand forecasting pipeline using Apache Spark and machine learning.  
The goal is to forecast daily retail demand by transforming raw transactional data into time-series features and evaluating multiple ML models.

The project follows real-world data engineering and ML pipeline practices with a focus on:
- Scalability
- Reproducibility
- Model evaluation
- Clean project structure

---

## 🧠 Problem Statement
Retail businesses need accurate demand forecasts to:
- Optimize inventory
- Reduce stock-outs and overstock
- Improve supply chain planning

Using historical transaction data, this project predicts **daily demand** based on lagged and rolling time-series features.

---

## 🛠 Tech Stack
- **Python**
- **Apache Spark (PySpark)**
- **Spark SQL**
- **Spark MLlib**
- **Pandas**
- **Scikit-learn**
- **Git & GitHub**

---
## 🏗 Project Architecture

1. Raw transactional retail data is ingested from Excel files using PySpark.
2. Data cleaning and validation are applied to remove invalid records.
3. Daily-level demand is aggregated using Spark SQL.
4. Time-series features such as lag values and rolling averages are engineered.
5. Machine learning models are trained using Spark MLlib.
6. Models are evaluated using RMSE and MAE.
7. Predictions are saved for downstream analysis.

retail-demand-forecasting/
│
├── data/
│ ├── raw/ # Raw input data (excluded from Git)
│ └── processed/ # Cleaned & aggregated datasets
│
├── notebooks/
│ ├── data_ingestion.ipynb
│ ├── 02_feature_engineering.ipynb
│ ├── 03_baseline_model.ipynb
│ ├── 04_advanced_model.ipynb
│ └── 05_model_evaluation_and_analysis.ipynb
│
├── sql/ # Spark SQL queries (optional)
├── src/ # Reusable helper scripts (future)
├── .gitignore
└── README.md

## 🔄 Pipeline Workflow
1. **Data Ingestion**
   - Load raw transactional data
   - Clean invalid records
   - Store cleaned data as Parquet

2. **Feature Engineering**
   - Daily demand aggregation
   - Lag features (lag_1, lag_7)
   - Rolling window features (7-day, 14-day averages)

3. **Baseline Model**
   - Linear Regression using Spark ML
   - Train-test split
   - RMSE & MAE evaluation

4. **Advanced Model**
   - RandomForest Regressor
   - Non-linear modeling
   - Performance comparison

5. **Model Evaluation**
   - Compare models using RMSE and MAE
   - Visual analysis and conclusions

---

## 📂 Notebook Execution Order

Run notebooks in the following order:

1. `data_ingestion.ipynb`  
   - Loads raw retail data
   - Performs cleaning and validation
   - Saves processed datasets

2. `02_feature_engineering.ipynb`  
   - Creates daily demand aggregates
   - Generates lag and rolling window features

3. `03_baseline_model.ipynb`  
   - Trains Linear Regression baseline model
   - Evaluates using RMSE and MAE

4. `04_advanced_model.ipynb`  
   - Trains RandomForest regression model
   - Saves predictions

5. `05_model_evaluation_and_analysis.ipynb`  
   - Compares model performance
   - Visualizes residuals


## 📊 Model Performance

| Model               | RMSE     | MAE      |
|--------------------|----------|----------|
| Linear Regression  | 12557.86 | 6839.22  |
| Random Forest      | 13432.02 | 7540.04  |

### ✅ Observation
Linear Regression outperformed RandomForest due to:
- Strong feature engineering
- Limited dataset size
- Reduced overfitting

---

### 1️⃣ Create virtual environment
```bash
python -m venv venv
source venv/bin/activate

## 🧠 Key Learnings

- Feature engineering has a larger impact than model complexity.
- Spark SQL is effective for large-scale aggregations.
- Simple models can outperform complex models on structured time-series data.
- Proper data validation is critical for demand forecasting accuracy.



