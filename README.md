# Retail Demand Forecasting using Apache Spark

## 📌 Overview
This project implements an end-to-end **retail demand forecasting pipeline** using **Apache Spark** and machine learning.
The objective is to forecast **daily retail demand** by transforming raw transactional data into time-series features and evaluating multiple models.

The project follows real-world **data engineering and ML pipeline practices** with a focus on:
- Scalability
- Reproducibility
- Model evaluation
- Clean project structure

---

## 🧠 Problem Statement
Retail businesses require accurate demand forecasts to:
- Optimize inventory levels
- Reduce stock-outs and overstock
- Improve supply chain planning

Using historical transaction data, this project predicts **daily demand** using lagged and rolling time-series features.

---

## 🛠 Tech Stack
- **Python**
- **Apache Spark (PySpark)**
- **Spark SQL**
- **Spark MLlib (Modeling)**
- **Pandas (Data export & analysis)**
- **PyTorch (Experimental deep learning model)**
- **Git & GitHub**

---

## 🏗 Project Architecture
1. Raw transactional data is ingested from Excel files using PySpark.
2. Data cleaning and validation remove invalid records.
3. Daily demand is aggregated using Spark SQL.
4. Time-series features (lags & rolling averages) are engineered.
5. Models are trained using Spark MLlib.
6. Models are evaluated using RMSE and MAE.
7. Predictions and metrics are saved for downstream analysis and visualization.

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
│ ├── 05_model_evaluation_and_analysis.ipynb
│ └── 06_experimental_pytorch_model.ipynb
│
├── spark-warehouse/ # Spark metadata (ignored in Git)
├── sql/ # Spark SQL queries (optional)
├── src/ # Reusable utilities (future)
├── .gitignore
└── README.md


---

## 🔄 Pipeline Workflow

### 1️⃣ Data Ingestion
- Load raw Excel retail data
- Standardize column names
- Remove invalid transactions
- Save cleaned data as Parquet

### 2️⃣ Feature Engineering
- Aggregate daily demand
- Generate time-based features
- Create lag features (`lag_1`, `lag_7`)
- Create rolling averages (`rolling_7`, `rolling_14`)

### 3️⃣ Baseline Model
- Spark MLlib **Linear Regression**
- Train / test split
- Evaluate with RMSE and MAE

### 4️⃣ Advanced Model
- Spark MLlib **RandomForest Regressor**
- Capture non-linear demand patterns
- Compare against baseline

### 5️⃣ Model Evaluation & Analysis
- Compare all models using RMSE and MAE
- Residual analysis
- Save comparison results for reporting

### 6️⃣ Experimental Model (Optional)
- PyTorch Neural Network
- In-memory validation
- Used for experimentation and learning

---

## 📂 Notebook Execution Order
Run notebooks in the following order:

1. **data_ingestion.ipynb**
   - Load and clean raw retail data
   - Save cleaned Parquet datasets

2. **02_feature_engineering.ipynb**
   - Aggregate daily demand
   - Create lag and rolling window features

3. **03_baseline_model.ipynb**
   - Train Spark Linear Regression model
   - Evaluate with RMSE and MAE

4. **04_advanced_model.ipynb**
   - Train Spark RandomForest model
   - Generate predictions

5. **05_model_evaluation_and_analysis.ipynb**
   - Compare Spark Linear Regression, RandomForest, and PyTorch models
   - Residual analysis and conclusions

6. **06_experimental_pytorch_model.ipynb**
   - Train a PyTorch neural network
   - Experimental comparison

---

## 📊 Model Performance

| Model                              | RMSE        | MAE        |
|-----------------------------------|-------------|------------|
| Spark Linear Regression            | 13191.14    | 7478.16    |
| Spark RandomForest                 | 12348.63    | 6313.76    |
| PyTorch Neural Network (Experimental) | 12795.53 | 8718.05    |

### ✅ Observations
- **Spark RandomForest** performed best overall.
- Feature engineering significantly impacted performance.
- PyTorch model demonstrates feasibility but requires more tuning and data.
- Distributed Spark models are more scalable for large datasets.

---

## ▶️ How to Run the Project

```bash
# Clone repository
git clone https://github.com/alapatilalitha/retail-demand-forecasting.git
cd retail-demand-forecastcasting

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
