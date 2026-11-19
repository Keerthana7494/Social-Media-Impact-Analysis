# Social Media Impact on Student Life – MLOps Project

A beginner-friendly end-to-end classification project using Python, scikit-learn, Streamlit and MLflow.

## 1. What this project does

1. Loads the supplied student social-media dataset.
2. Checks missing values, duplicates, data types and feature ranges.
3. Cleans text and numeric fields; invalid domain values become missing.
4. Uses median/mode imputers inside a reusable scikit-learn pipeline.
5. Creates simple features:
   - Total_Usage_Hours
   - Usage_to_Sleep_Ratio
   - Late_Night_Usage_Hours_Flag
6. Visualizes numerical feature distributions, target balance and correlations.
7. Compares Logistic Regression, Random Forest and Extra Trees.
8. Selects the model using Macro F1 (with Balanced Accuracy as a tie-breaker).
9. Evaluates the selected model with a classification report and confusion matrix.
10. Tracks each experiment in MLflow.
11. Saves versioned models and reports.
12. Provides a Streamlit dashboard and a prediction form.

## 2. Project structure

```text
social_media_mlops_project/
├── app.py
├── requirements.txt
├── README.md
├── data/
│   └── Social_media_impact_on_life.csv
├── src/
│   ├── __init__.py
│   ├── config.py
│   ├── data.py
│   ├── visualizations.py
│   └── train.py
├── models/
│   ├── model_v1.joblib
│   ├── model_v1_metadata.json
│   └── latest_model.json
├── reports/
│   ├── cleaned_dataset.csv
│   ├── data_cleaning_report.json
│   ├── model_comparison.csv
│   ├── classification_report.json
│   ├── test_predictions.csv
│   └── figures/
└── mlruns/ or mlflow.db
```

The `models`, `reports`, and MLflow tracking files are created when you run the pipeline.

## 3. Requirements

- Python 3.10 or newer
- VS Code or another Python IDE
- Internet connection for installing packages (not needed after installation)

## 4. Windows setup (VS Code)

Open the project folder in VS Code. Open Terminal and run:

### Step 1 – Create virtual environment

```bash
python -m venv .venv
```

### Step 2 – Activate it

PowerShell:
```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:
```cmd
.venv\Scripts\activate.bat
```

### Step 3 – Install libraries

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4 – Run the Streamlit application

```bash
streamlit run app.py
```

The browser opens the dashboard. Go to **Train Models** and click **Run complete training pipeline**.

## 5. Run training without Streamlit

From the project root:

```bash
python -m src.train
```

## 6. View MLflow experiments

Open a second terminal in the project folder, activate the same virtual environment, then run:

```bash
mlflow ui
```

Open:

http://localhost:5000

Each candidate model is logged as a separate MLflow run with parameters, metrics and its fitted pipeline.

## 7. Reports generated

| Output | Purpose |
|---|---|
| cleaned_dataset.csv | Cleaned source records |
| data_cleaning_report.json | Rows removed and cleaning summary |
| numeric_feature_distributions.png | Feature distribution charts |
| target_distribution.png | Class imbalance visualization |
| correlation_heatmap.png | Numerical feature correlations |
| model_comparison.csv | Model metrics comparison |
| model_comparison.png | Macro F1 comparison chart |
| classification_report.json | Precision, recall and F1 by class |
| confusion_matrix.png | Correct and incorrect predictions |
| test_predictions.csv | Actual vs predicted test labels |

## 8. Why Macro F1?

The target has imbalanced classes. Accuracy alone can look high when a model mainly predicts the majority class. Macro F1 calculates F1 for each class and gives every class equal weight. This makes it a more useful primary selection metric for this dataset.

The best model is selected from the test-set comparison in this educational project. For a stricter production evaluation, use a validation set or cross-validation for model selection and reserve a final untouched test set for one-time evaluation.

## 9. Model versioning

Every training run saves the next available version:

```text
models/model_v1.joblib
models/model_v2.joblib
models/model_v3.joblib
```

`latest_model.json` points to the latest saved version. Metadata records the model name, training time, target, feature names and Macro F1.

## 10. Important notes

- Missing feature values are imputed by the fitted preprocessing pipeline. This prevents the model from receiving null feature values.
- Rows with missing target labels cannot be used for supervised learning and are excluded.
- Extreme but valid values are not automatically deleted.
- `Student_ID` is excluded from model training.
- The dataset is observational. Predictions show learned associations, not proof that social media causes a particular mental-health or academic outcome.
- Do not use this model to make medical or high-stakes decisions.
- Do not upload real student personal data to a public repository.

## 11. Resume description

**Social Media Impact on Student Life – End-to-End MLOps Classification**

Built an end-to-end machine learning application to classify the overall impact of social media usage on students. Implemented data validation, preprocessing, feature engineering, exploratory visualizations, model benchmarking, evaluation, MLflow experiment tracking, versioned model persistence and an interactive Streamlit dashboard using Python and scikit-learn.

**Technologies:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, MLflow, Streamlit, Joblib, Git.
