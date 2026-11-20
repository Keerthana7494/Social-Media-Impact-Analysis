
import json
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
import matplotlib.pyplot as plt

from datetime import datetime

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from src.config import (
    TARGET,
    ID_COLUMN,
    RANDOM_STATE,
    TEST_SIZE,
    REPORTS_DIR,
    FIGURES_DIR,
    MODELS_DIR
)

from src.data import (
    load_data,
    clean_data,
    engineer_features,
    prepare_features
)

from src.visualizations import (
    save_data_visuals,
    save_confusion_matrix,
    save_validation_before_after_chart,
    save_hyperparameter_search_chart,
    save_final_test_metrics_chart
)


# ============================================================
# CONFIGURATION
# ============================================================

VALIDATION_FRACTION = 0.20

MLFLOW_EXPERIMENT = "Social_Media_Impact_Classification"


# ============================================================
# 1. BUILD PREPROCESSING + MODEL PIPELINE
# ============================================================

def build_pipeline(X, model):
    """
    Build a complete ML pipeline.

    Numerical features:
        Missing values -> Median
        Scaling -> StandardScaler

    Categorical features:
        Missing values -> Most frequent
        Encoding -> OneHotEncoder

    The preprocessing steps are fitted only on the
    training data when pipeline.fit() is called.
    """

    # Identify numerical columns
    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    # Remaining columns are treated as categorical.
    # This includes text, category and boolean columns.
    categorical_features = [
        column
        for column in X.columns
        if column not in numeric_features
    ]

    transformers = []

    # Numerical preprocessing
    if numeric_features:

        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median",
                        keep_empty_features=True
                    )
                ),
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )
        )

    # Categorical preprocessing
    if categorical_features:

        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                        keep_empty_features=True
                    )
                ),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore"
                    )
                )
            ]
        )

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )

    if not transformers:
        raise ValueError(
            "No valid features were found for model training."
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", model)
        ]
    )

    return pipeline


# ============================================================
# 2. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

def split_dataset(X, y):
    """
    Split dataset into:

        Training    = 64%
        Validation  = 16%
        Testing     = 20%

    TEST_SIZE controls the final test percentage.

    Stratification preserves target class distribution.
    """

    if not 0 < TEST_SIZE < 1:
        raise ValueError(
            "TEST_SIZE must be between 0 and 1."
        )

    if not 0 < VALIDATION_FRACTION < 1:
        raise ValueError(
            "VALIDATION_FRACTION must be between 0 and 1."
        )

    class_counts = y.value_counts()

    if class_counts.min() < 3:
        raise ValueError(
            "Each target class should contain at least "
            "3 records for a stratified train/validation/test split.\n"
            f"Current class counts:\n{class_counts.to_string()}"
        )

    try:

        # Step 1: Separate final test dataset
        X_train_val, X_test, y_train_val, y_test = (
            train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y
            )
        )

        # Step 2: Separate validation dataset
        X_train, X_val, y_train, y_val = (
            train_test_split(
                X_train_val,
                y_train_val,
                test_size=VALIDATION_FRACTION,
                random_state=RANDOM_STATE,
                stratify=y_train_val
            )
        )

    except ValueError as error:

        raise ValueError(
            "Stratified splitting failed. "
            "The dataset may be too small or some target "
            "classes may have insufficient records.\n"
            f"Original error: {error}"
        ) from error

    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        X_train_val,
        y_train_val
    )


# ============================================================
# 3. CREATE CLEANED DATASET REPORT
# ============================================================

def create_cleaned_report(
    featured_data,
    X_train_val,
    feature_columns
):
    """
    Create an imputed CSV for reporting purposes.

    Numerical columns:
        Training + validation median

    Categorical columns:
        Training + validation mode

    The final test dataset is not used to calculate
    imputation statistics.

    Important:
        The model pipeline performs its own imputation.
        This function only creates the reporting CSV.
    """

    report_data = featured_data.copy()

    missing_before = int(
        report_data[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    print("\nREPORT IMPUTATION DETAILS")
    print("-------------------------")

    for column in feature_columns:

        if report_data[column].isna().sum() == 0:
            continue

        training_column = X_train_val[column]

        # Numerical columns
        if pd.api.types.is_numeric_dtype(training_column) and not (
            pd.api.types.is_bool_dtype(training_column)
        ):

            median_value = training_column.median()

            if pd.notna(median_value):

                report_data[column] = (
                    report_data[column].fillna(median_value)
                )

                print(
                    f"{column}: Filled with training median "
                    f"({median_value})"
                )

        # Categorical and Boolean columns
        else:

            mode_values = training_column.mode(
                dropna=True
            )

            if not mode_values.empty:

                mode_value = mode_values.iloc[0]

                report_data[column] = (
                    report_data[column].fillna(mode_value)
                )

                print(
                    f"{column}: Filled with training mode "
                    f"({mode_value})"
                )

    missing_after = int(
        report_data[feature_columns]
        .isna()
        .sum()
        .sum()
    )

    print("\nCLEANED DATASET SUMMARY")
    print("-----------------------")

    print("Missing values before:", missing_before)
    print("Missing values after:", missing_after)

    return (
        report_data,
        missing_before,
        missing_after
    )


# ============================================================
# 4. MODEL EVALUATION METRICS
# ============================================================

def calculate_metrics(y_true, predictions):
    """
    Calculate classification evaluation metrics.
    """

    return {

        "accuracy": accuracy_score(
            y_true,
            predictions
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            predictions
        ),

        "macro_f1": f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        ),

        "weighted_f1": f1_score(
            y_true,
            predictions,
            average="weighted",
            zero_division=0
        )

    }


# ============================================================
# 5. SAVE MODEL COMPARISON CHART
# ============================================================

def save_model_comparison_chart(results_df):
    """
    Visualize validation Macro F1 scores.
    """

    figures_dir = REPORTS_DIR / "figures"

    figures_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    plt.figure(figsize=(10, 6))

    bars = plt.bar(
        results_df["model"],
        results_df["macro_f1"]
    )

    # Display score above each bar
    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.3f}",
            ha="center",
            va="bottom"
        )

    plt.title(
        "Model Comparison - Validation Macro F1"
    )

    plt.xlabel("Machine Learning Models")

    plt.ylabel("Macro F1 Score")

    plt.ylim(0, 1.10)

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )

    plt.tight_layout()

    chart_path = (
        figures_dir / "model_comparison.png"
    )

    plt.savefig(
        chart_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "Model comparison chart saved:",
        chart_path
    )

    return chart_path


# ============================================================
# 5A. TRAINING VS TESTING PERFORMANCE / OVERFITTING CHECK
# ============================================================

def save_train_test_comparison_chart(train_metrics, test_metrics):
    """Save a chart comparing fit-data scores with unseen test scores."""

    figures_dir = REPORTS_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    metric_names = ["Accuracy", "Macro F1"]
    train_scores = [train_metrics["accuracy"], train_metrics["macro_f1"]]
    test_scores = [test_metrics["accuracy"], test_metrics["macro_f1"]]

    x_positions = list(range(len(metric_names)))
    bar_width = 0.35

    plt.figure(figsize=(9, 5))

    train_bars = plt.bar(
        [x - bar_width / 2 for x in x_positions],
        train_scores,
        bar_width,
        label="Training + Validation"
    )
    test_bars = plt.bar(
        [x + bar_width / 2 for x in x_positions],
        test_scores,
        bar_width,
        label="Unseen Test"
    )

    for bars in (train_bars, test_bars):
        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.015,
                f"{height:.3f}",
                ha="center",
                va="bottom"
            )

    plt.xticks(x_positions, metric_names)
    plt.ylim(0, 1.12)
    plt.ylabel("Score")
    plt.title("Final Model: Training vs Test Performance")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.tight_layout()

    chart_path = figures_dir / "train_test_comparison.png"
    plt.savefig(chart_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("Train/test comparison chart saved:", chart_path)
    return chart_path


def interpret_generalization_gap(train_metrics, test_metrics):
    """
    Calculate the Macro F1 train-test gap.
    Thresholds are practical warning indicators, not universal rules.
    """

    gap = train_metrics["macro_f1"] - test_metrics["macro_f1"]

    if gap > 0.10:
        interpretation = "Possible overfitting: large training-test gap."
    elif gap > 0.05:
        interpretation = "Warning: moderate training-test gap."
    elif gap >= 0:
        interpretation = "No large training-test gap detected."
    else:
        interpretation = (
            "Test score is higher than training score; "
            "check dataset size and split variation."
        )

    return float(gap), interpretation


# ============================================================
# 6. SAVE JSON REPORT
# ============================================================

def save_json_report(data, file_path):
    """
    Save a JSON report.

    default=str handles values such as NumPy types
    or timestamps that are not directly JSON serializable.
    """

    with open(
        file_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4,
            default=str
        )


# ============================================================
# 7. DEFINE MACHINE LEARNING MODELS
# ============================================================

def get_models():

    return {

        # Baseline model
        "Dummy Classifier": DummyClassifier(
            strategy="most_frequent"
        ),

        # Linear model
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        # Decision tree
        "Decision Tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE
        ),

        # Random forest
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_split=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1
        ),

        # Gradient boosting
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=RANDOM_STATE
        )

    }


# ============================================================
# 8. TRAIN AND COMPARE MODELS
# ============================================================

def train_and_compare_models(
    X_train,
    y_train,
    X_val,
    y_val,
    feature_count
):
    """
    Train all candidate models using training data.

    Evaluate each candidate using validation data.

    Return:
        results_df
        trained_models
    """

    models = get_models()

    results = []

    trained_models = {}

    print("\nSTARTING MODEL TRAINING")
    print("-----------------------")

    for model_name, estimator in models.items():

        print(f"\nTraining: {model_name}")

        pipeline = build_pipeline(
            X_train,
            clone(estimator)
        )

        with mlflow.start_run(
            run_name=model_name
        ):

            # Train only on training data
            pipeline.fit(
                X_train,
                y_train
            )

            # Validate on validation data
            predictions = pipeline.predict(
                X_val
            )

            metrics = calculate_metrics(
                y_val,
                predictions
            )

            # Log parameters
            mlflow.log_param(
                "model_name",
                model_name
            )

            mlflow.log_param(
                "random_state",
                RANDOM_STATE
            )

            mlflow.log_param(
                "test_size",
                TEST_SIZE
            )

            mlflow.log_param(
                "validation_fraction",
                VALIDATION_FRACTION
            )

            mlflow.log_param(
                "feature_count",
                feature_count
            )

            mlflow.log_param(
                "training_records",
                len(X_train)
            )

            mlflow.log_param(
                "validation_records",
                len(X_val)
            )

            # Log validation metrics
            mlflow.log_metrics(metrics)

            # Log trained candidate model
            mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                skops_trusted_types=["numpy.dtype","sklearn.tree._tree.Tree"]
                #serialization_format="pickle"
            )

            results.append({

                "model": model_name,

                **metrics

            })

            trained_models[model_name] = estimator

            print(
                "Validation metrics:",
                metrics
            )

    results_df = pd.DataFrame(results)

    # Highest Macro F1 first.
    # Balanced Accuracy is the tie-breaker.
    results_df = results_df.sort_values(
        by=[
            "macro_f1",
            "balanced_accuracy"
        ],
        ascending=False
    ).reset_index(drop=True)

    return (
        results_df,
        trained_models
    )


# ============================================================
# 8A. HYPERPARAMETER SEARCH CONFIGURATION
# ============================================================

def get_parameter_grid(model_name):
    """Return a modest, model-specific GridSearchCV parameter grid."""
    
    grids = {
    "Logistic Regression": {
        "classifier__C": [0.1, 1.0, 10.0],
        "classifier__solver": ["lbfgs"],
        "classifier__class_weight": [None, "balanced"],
    },

    "Decision Tree": {
        "classifier__max_depth": [5, 10, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__class_weight": [None, "balanced"],
    },

    "Random Forest": {
        "classifier__n_estimators": [150, 250],
        "classifier__max_depth": [None, 10, 20],
        "classifier__min_samples_split": [2, 5],
        "classifier__min_samples_leaf": [1, 2, 4],
        "classifier__class_weight": [None, "balanced"],
    },

    "Gradient Boosting": {
    "classifier__n_estimators": [50, 100, 150],
    "classifier__learning_rate": [0.01, 0.03, 0.05],
    "classifier__max_depth": [1, 2, 3],
    "classifier__min_samples_split": [5, 10, 15],
    "classifier__min_samples_leaf": [3, 5, 8],
    "classifier__subsample": [0.7, 0.8, 0.9],
    "classifier__max_features": ["sqrt", None],
},

    "Dummy Classifier": {
        "classifier__strategy": [
            "most_frequent",
            "prior",
            "stratified",
        ],
    },


}
    if model_name not in grids:
        raise ValueError(f"No hyperparameter grid configured for {model_name}.")
    return grids[model_name]


def tune_selected_model(model_name, estimator, X_train, y_train, X_val, y_val):
    """Tune only the validation-selected model; CV uses training data only."""
    print("\n========== HYPERPARAMETER TUNING ==========")
    print("Selected model:", model_name)

    pipeline = build_pipeline(X_train, clone(estimator))
    parameter_grid = get_parameter_grid(model_name)

    min_class_count = int(y_train.value_counts().min())
    cv_splits = min(3, min_class_count)
    if cv_splits < 2:
        raise ValueError("Not enough samples in every training class for stratified CV tuning.")

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        refit=True,
        verbose=1,
        return_train_score=True,
        error_score="raise",
    )

    search.fit(X_train, y_train)
    tuned_model = search.best_estimator_
    tuned_predictions = tuned_model.predict(X_val)
    tuned_validation_metrics = calculate_metrics(y_val, tuned_predictions)

    print("\nBEST HYPERPARAMETERS")
    print(json.dumps(search.best_params_, indent=4, default=str))
    print(f"Best cross-validation Macro F1: {search.best_score_:.4f}")
    print("Tuned validation metrics:", tuned_validation_metrics)

    cv_results = pd.DataFrame(search.cv_results_)
    cv_results.to_csv(REPORTS_DIR / "hyperparameter_tuning_results.csv", index=False)

    best_parameters = {
        "selected_model": model_name,
        "best_cv_macro_f1": float(search.best_score_),
        "best_parameters": search.best_params_,
        "cv_folds": cv_splits,
        "scoring": "f1_macro",
    }
    save_json_report(best_parameters, REPORTS_DIR / "best_hyperparameters.json")

    return tuned_model, tuned_validation_metrics, search, cv_results


# ============================================================
# 9. MAIN MLOPS TRAINING PIPELINE
# ============================================================

def run_training(
    data_path,
    experiment_name=MLFLOW_EXPERIMENT
):

    print("\n========== MLOPS PIPELINE STARTED ==========\n")

    # --------------------------------------------------------
    # CREATE OUTPUT DIRECTORIES
    # --------------------------------------------------------

    REPORTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    (REPORTS_DIR / "figures").mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # STEP 1: LOAD DATA
    # --------------------------------------------------------

    print("STEP 1: Loading dataset...")

    raw_data = load_data(data_path)

    print(
        "Original dataset shape:",
        raw_data.shape
    )

    # --------------------------------------------------------
    # STEP 2: CLEAN DATA
    # --------------------------------------------------------

    print("\nSTEP 2: Cleaning dataset...")

    cleaned_data, cleaning_report = clean_data(
        raw_data
    )

    print(
        "Cleaned dataset shape:",
        cleaned_data.shape
    )

    print(
        "Missing values after basic cleaning:",
        cleaned_data.isna().sum().sum()
    )

    # --------------------------------------------------------
    # STEP 3: FEATURE ENGINEERING
    # --------------------------------------------------------

    print("\nSTEP 3: Feature engineering...")

    featured_data = engineer_features(
        cleaned_data
    )

    print(
        "Dataset shape after feature engineering:",
        featured_data.shape
    )

    # --------------------------------------------------------
    # STEP 4: PREPARE FEATURES AND TARGET
    # --------------------------------------------------------

    print("\nSTEP 4: Preparing features and target...")

    # IMPORTANT:
    # Use prepare_features() as the single source of truth.
    # Do not overwrite X and y using manual drop operations.
    #
    # This also ensures that ID columns and unapproved
    # columns are excluded according to data.py.

    X, y = prepare_features(
        featured_data
    )

    y = y.astype(str)

    if X.empty or X.shape[1] == 0:

        raise ValueError(
            "No features are available for model training."
        )

    if len(X) != len(y):

        raise ValueError(
            "Feature and target record counts do not match."
        )

    if y.isna().any():

        raise ValueError(
            "Target column contains missing values."
        )

    if y.nunique() < 2:

        raise ValueError(
            "At least two target classes are required."
        )

    print("Feature count:", X.shape[1])

    print("Feature columns:", X.columns.tolist())

    print("Target:", TARGET)

    print("Target classes:", y.unique().tolist())

    print("\nTarget distribution:")

    print(
        y.value_counts().to_string()
    )

    # --------------------------------------------------------
    # STEP 5: TRAIN / VALIDATION / TEST SPLIT
    # --------------------------------------------------------

    print("\nSTEP 5: Splitting dataset...")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        X_train_val,
        y_train_val
    ) = split_dataset(
        X,
        y
    )

    print("\nDATASET SPLIT SUMMARY")
    print("---------------------")

    print(
        f"Training records:   {len(X_train)} "
        f"({len(X_train) / len(X) * 100:.2f}%)"
    )

    print(
        f"Validation records: {len(X_val)} "
        f"({len(X_val) / len(X) * 100:.2f}%)"
    )

    print(
        f"Testing records:    {len(X_test)} "
        f"({len(X_test) / len(X) * 100:.2f}%)"
    )

    # --------------------------------------------------------
    # STEP 6: CREATE CLEANED DATASET REPORT
    # --------------------------------------------------------

    print("\nSTEP 6: Creating cleaned dataset report...")

    (
        cleaned_report,
        missing_before,
        missing_after
    ) = create_cleaned_report(
        featured_data,
        X_train_val,
        X.columns.tolist()
    )

    cleaned_report_path = (
        REPORTS_DIR / "cleaned_dataset.csv"
    )

    cleaned_report.to_csv(
        cleaned_report_path,
        index=False
    )

    cleaning_report["missing_values_before_imputation"] = (
        missing_before
    )

    cleaning_report["missing_values_after_imputation"] = (
        missing_after
    )

    cleaning_report["imputation_method"] = {

        "numerical": "Median",

        "categorical": "Most Frequent",

        "statistics_source": "Training + Validation data only",

        "model_imputation": (
            "Performed independently inside the sklearn pipeline"
        )

    }

    cleaning_report_path = (
        REPORTS_DIR / "data_cleaning_report.json"
    )

    save_json_report(
        cleaning_report,
        cleaning_report_path
    )

    remaining_missing = (
        cleaned_report[X.columns]
        .isna()
        .sum()
    )

    print("\nRemaining missing values:")

    print(
        remaining_missing[
            remaining_missing > 0
        ].to_string()
    )

    if remaining_missing.sum() > 0:

        print(
            "\nWARNING: Some feature columns still contain "
            "missing values. Check whether their training "
            "values are entirely missing."
        )

    else:

        print(
            "\nSUCCESS: All feature missing values handled "
            "in the reporting dataset."
        )

    # --------------------------------------------------------
    # STEP 7: EXPLORATORY DATA ANALYSIS
    # --------------------------------------------------------

    print("\nSTEP 7: Generating EDA visualizations...")

    save_data_visuals(
        featured_data
    )

    # --------------------------------------------------------
    # STEP 8: START MLFLOW EXPERIMENT
    # --------------------------------------------------------

    print("\nSTEP 8: Configuring MLflow...")

    mlflow.set_experiment(
        experiment_name
    )

    # --------------------------------------------------------
    # STEP 9: TRAIN AND COMPARE MODELS
    # --------------------------------------------------------

    print("\nSTEP 9: Training and comparing models...")

    (
        results_df,
        trained_models
    ) = train_and_compare_models(
        X_train,
        y_train,
        X_val,
        y_val,
        X.shape[1]
    )

    # --------------------------------------------------------
    # STEP 10: SAVE MODEL COMPARISON
    # --------------------------------------------------------

    print("\nSTEP 10: Saving model comparison...")

    comparison_path = (
        REPORTS_DIR / "model_comparison.csv"
    )

    results_df.to_csv(
        comparison_path,
        index=False
    )

    print("\nMODEL COMPARISON - VALIDATION RESULTS")

    print(
        results_df.to_string(index=False)
    )

    print(
        "\nModel comparison CSV saved:",
        comparison_path
    )

    # Use the locally defined chart function.
    comparison_chart_path = (
        save_model_comparison_chart(
            results_df
        )
    )

    # --------------------------------------------------------
    # STEP 11: SELECT BEST MODEL
    # --------------------------------------------------------

    print("\nSTEP 11: Selecting best model...")

    best_name = results_df.iloc[0]["model"]

    best_estimator = trained_models[best_name]

    best_validation_metrics = {

        key: float(value)

        for key, value in results_df.iloc[0].items()

        if key != "model"

    }

    print("Selected model:", best_name)

    print(
        "Selection metric: Macro F1"
    )

    print(
        "Tie-breaker: Balanced Accuracy"
    )

    print(
        "Best validation metrics:",
        best_validation_metrics
    )

    # --------------------------------------------------------
    # STEP 12: HYPERPARAMETER TUNING OF SELECTED MODEL
    # --------------------------------------------------------

    print("\nSTEP 12: Hyperparameter tuning of selected model...")

    tuned_model, tuned_validation_metrics, search, tuning_results_df = (
        tune_selected_model(
            best_name,
            best_estimator,
            X_train,
            y_train,
            X_val,
            y_val
        )
    )

    print("\nORIGINAL VS TUNED MODEL - VALIDATION RESULTS")
    before_after_df = pd.DataFrame([
        {"model_version": "Original", **best_validation_metrics},
        {"model_version": "Tuned", **tuned_validation_metrics},
    ])
    print(before_after_df.to_string(index=False))
    before_after_path = REPORTS_DIR / "validation_before_after_tuning.csv"
    before_after_df.to_csv(before_after_path, index=False)

    tuning_chart_path = save_hyperparameter_search_chart(tuning_results_df)
    before_after_chart_path = save_validation_before_after_chart(before_after_df)

    print("\nTuning results saved:", REPORTS_DIR / "hyperparameter_tuning_results.csv")
    print("Best parameters saved:", REPORTS_DIR / "best_hyperparameters.json")
    print("Tuning chart saved:", tuning_chart_path)
    print("Before/after chart saved:", before_after_chart_path)

    # --------------------------------------------------------
    # STEP 13: RETRAIN TUNED MODEL
    # --------------------------------------------------------

    print("\nSTEP 12: Retraining best model...")

    # Combine training + validation datasets.
    # The final test dataset is not included.

    # Rebuild the selected tuned estimator on all training + validation data.
    tuned_estimator = clone(tuned_model.named_steps["classifier"])
    final_model = build_pipeline(
        X_train_val,
        tuned_estimator
    )

    final_model.fit(
        X_train_val,
        y_train_val
    )

    print(
        "Final model trained using training + validation data."
    )

    # --------------------------------------------------------
    # STEP 13: TRAINING + VALIDATION AND FINAL TEST EVALUATION
    # --------------------------------------------------------

    print("\nSTEP 13: Evaluating final model...")

    # This is the same combined dataset used to fit the final model.
    train_predictions = final_model.predict(X_train_val)
    train_metrics = calculate_metrics(y_train_val, train_predictions)

    # The test dataset remains unseen during fitting and tuning.
    best_predictions = final_model.predict(X_test)
    final_metrics = calculate_metrics(y_test, best_predictions)

    generalization_gap, overfitting_interpretation = (
        interpret_generalization_gap(train_metrics, final_metrics)
    )

    train_test_chart_path = save_train_test_comparison_chart(
        train_metrics,
        final_metrics
    )
    final_test_chart_path = save_final_test_metrics_chart(final_metrics)

    print("\nTRAINING + VALIDATION METRICS")
    print("-----------------------------")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nFINAL TEST METRICS")
    print("------------------")
    for metric, value in final_metrics.items():
        print(f"{metric}: {value:.4f}")

    print("\nOVERFITTING CHECK")
    print("-----------------")
    print(f"Training Macro F1: {train_metrics['macro_f1']:.4f}")
    print(f"Test Macro F1:     {final_metrics['macro_f1']:.4f}")
    print(f"Generalization gap: {generalization_gap:.4f}")
    print("Interpretation:", overfitting_interpretation)

    # Classification report
    labels = sorted(
        y.unique().tolist()
    )

    report = classification_report(
        y_test,
        best_predictions,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    classification_report_path = (
        REPORTS_DIR / "classification_report.json"
    )

    save_json_report(
        report,
        classification_report_path
    )

    # Confusion matrix
    matrix = confusion_matrix(
        y_test,
        best_predictions,
        labels=labels
    )

    save_confusion_matrix(
        matrix,
        labels
    )

    # Actual vs predicted values
    prediction_output = X_test.copy()

    prediction_output["actual"] = (
        y_test.values
    )

    prediction_output["predicted"] = (
        best_predictions
    )

    prediction_path = (
        REPORTS_DIR / "test_predictions.csv"
    )

    prediction_output.to_csv(
        prediction_path,
        index=False
    )

    # --------------------------------------------------------
    # STEP 14: SAVE FINAL MODEL METRICS
    # --------------------------------------------------------

    print("\nSTEP 14: Saving final evaluation reports...")

    final_metrics_output = {

        "selected_model": best_name,

        "selection_metric": "Macro F1",

        "tie_breaker": "Balanced Accuracy",

        "validation_metrics": best_validation_metrics,

        "tuned_validation_metrics": tuned_validation_metrics,

        "training_plus_validation_metrics": {
            key: float(value)
            for key, value in train_metrics.items()
        },

        "generalization_gap_macro_f1": generalization_gap,

        "overfitting_interpretation": overfitting_interpretation,

        "best_hyperparameters": search.best_params_,

        "cross_validation_macro_f1": float(search.best_score_),

        "test_metrics": {

            key: float(value)

            for key, value in final_metrics.items()

        }

    }

    best_metrics_path = (
        REPORTS_DIR / "best_model_metrics.json"
    )

    save_json_report(
        final_metrics_output,
        best_metrics_path
    )

    # --------------------------------------------------------
    # STEP 15: LOG FINAL MODEL IN MLFLOW
    # --------------------------------------------------------

    print("\nSTEP 15: Logging final model in MLflow...")

    with mlflow.start_run(
        run_name=f"Final_{best_name}"
    ):

        mlflow.log_param(
            "selected_model",
            best_name
        )

        mlflow.log_param("hyperparameter_tuning", "GridSearchCV")
        mlflow.log_param("cv_folds", search.n_splits_ if hasattr(search, "n_splits_") else 3)
        mlflow.log_params({f"tuned_{key}": str(value) for key, value in search.best_params_.items()})
        mlflow.log_metric("best_cv_macro_f1", float(search.best_score_))

        mlflow.log_param(
            "training_strategy",
            "Training + Validation"
        )

        mlflow.log_param(
            "selection_metric",
            "Macro F1"
        )

        mlflow.log_param(
            "tie_breaker",
            "Balanced Accuracy"
        )

        mlflow.log_param(
            "feature_count",
            X.shape[1]
        )

        mlflow.log_param(
            "training_records",
            len(X_train_val)
        )

        mlflow.log_param(
            "test_records",
            len(X_test)
        )

        # Log fit-data and test metrics separately in MLflow.
        mlflow.log_metrics({
            f"train_validation_{metric}": float(value)
            for metric, value in train_metrics.items()
        })

        mlflow.log_metrics({
            f"test_{metric}": float(value)
            for metric, value in final_metrics.items()
        })

        mlflow.log_metric(
            "generalization_gap_macro_f1",
            generalization_gap
        )

        # Log the final retrained model.
        mlflow.sklearn.log_model(
            sk_model=final_model,
            name="final_model",
            skops_trusted_types=["numpy.dtype","sklearn.tree._tree.Tree"]
        )

        # Log reports and visualizations as MLflow artifacts.
        artifact_paths = [

            comparison_path,

            comparison_chart_path,

            tuning_chart_path,

            before_after_chart_path,

            before_after_path,

            REPORTS_DIR / "hyperparameter_tuning_results.csv",

            REPORTS_DIR / "best_hyperparameters.json",

            final_test_chart_path,

            train_test_chart_path,

            FIGURES_DIR / "confusion_matrix.png",

            cleaned_report_path,

            cleaning_report_path,

            classification_report_path,

            prediction_path,

            best_metrics_path

        ]

        for artifact_path in artifact_paths:

            if artifact_path.exists():

                mlflow.log_artifact(
                    str(artifact_path)
                )

    # --------------------------------------------------------
    # STEP 16: SAVE AND VERSION FINAL MODEL
    # --------------------------------------------------------

    print("\nSTEP 16: Saving and versioning final model...")

    existing_models = list(
        MODELS_DIR.glob("model_v*.joblib")
    )

    versions = [

        int(path.stem.split("_v")[-1])

        for path in existing_models

        if path.stem.split("_v")[-1].isdigit()

    ]

    version = max(
        versions or [0]
    ) + 1

    model_path = (
        MODELS_DIR / f"model_v{version}.joblib"
    )

    joblib.dump(
        final_model,
        model_path
    )

    # --------------------------------------------------------
    # STEP 17: SAVE MODEL METADATA
    # --------------------------------------------------------

    print("\nSTEP 17: Saving model metadata...")

    metadata = {

        "model_version": version,

        "model_name": best_name,

        "hyperparameter_tuning": "GridSearchCV",

        "best_hyperparameters": search.best_params_,

        "cross_validation_macro_f1": float(search.best_score_),

        "target": TARGET,

        "accuracy": float(
            final_metrics["accuracy"]
        ),

        "macro_f1": float(
            final_metrics["macro_f1"]
        ),

        "balanced_accuracy": float(
            final_metrics["balanced_accuracy"]
        ),

        "weighted_f1": float(
            final_metrics["weighted_f1"]
        ),

        "validation_macro_f1": float(
            best_validation_metrics["macro_f1"]
        ),

        "training_plus_validation_macro_f1": float(
            train_metrics["macro_f1"]
        ),

        "generalization_gap_macro_f1": generalization_gap,

        "overfitting_interpretation": overfitting_interpretation,

        "trained_at": datetime.now().isoformat(
            timespec="seconds"
        ),

        "feature_columns": X.columns.tolist(),

        "training_records": len(X_train_val),

        "validation_records": len(X_val),

        "test_records": len(X_test)

    }

    metadata_path = (
        MODELS_DIR / f"model_v{version}_metadata.json"
    )

    save_json_report(
        metadata,
        metadata_path
    )

    # --------------------------------------------------------
    # STEP 18: UPDATE LATEST MODEL REFERENCE
    # --------------------------------------------------------

    latest_model = {

        "model_path": str(
            model_path.relative_to(
                MODELS_DIR.parent
            )
        ),

        "metadata_path": str(
            metadata_path.relative_to(
                MODELS_DIR.parent
            )
        )

    }

    latest_model_path = (
        MODELS_DIR / "latest_model.json"
    )

    save_json_report(
        latest_model,
        latest_model_path
    )

    # --------------------------------------------------------
    # STEP 19: RETURN RESULTS
    # --------------------------------------------------------

    print("\n========== MLOPS PIPELINE COMPLETED ==========\n")

    print("Best model:", best_name)

    print("Model version:", version)

    print("Saved model:", model_path)

    print("Model comparison CSV:", comparison_path)

    print("Model comparison chart:", comparison_chart_path)
    print("Hyperparameter tuning chart:", tuning_chart_path)
    print("Before/after chart:", before_after_chart_path)
    print("Final test metrics chart:", final_test_chart_path)
    print("Best hyperparameters:", search.best_params_)
    print("Final test metrics:", final_metrics)

    print("Reports folder:", REPORTS_DIR)

    return {

        "cleaning_report": cleaning_report,

        "comparison": results_df,

        "best_model": best_name,

        "validation_metrics": best_validation_metrics,

        "tuned_validation_metrics": tuned_validation_metrics,

        "best_hyperparameters": search.best_params_,

        "cross_validation_macro_f1": float(search.best_score_),

        "tuning_results": tuning_results_df,

        "before_after_tuning": before_after_df,

        "best_metrics": final_metrics,

        "training_metrics": train_metrics,

        "generalization_gap": generalization_gap,

        "overfitting_interpretation": overfitting_interpretation,

        "train_test_chart": str(train_test_chart_path),

        "model_path": str(model_path),

        "model_version": version,

        "report": report,

        "comparison_chart": str(comparison_chart_path),

        "tuning_chart": str(tuning_chart_path),

        "before_after_chart": str(before_after_chart_path),

        "final_test_chart": str(final_test_chart_path)

    }


# ============================================================
# 20. EXECUTE PIPELINE DIRECTLY
# ============================================================

if __name__ == "__main__":

    from src.config import DATA_DIR

    result = run_training(
        DATA_DIR / "Social_media_impact_on_life.csv"
    )

    print("\nFINAL MODEL COMPARISON")

    print(
        result["comparison"].to_string(
            index=False
        )
    )

    print(
        "\nBest model:",
        result["best_model"]
    )

    print(
        "Final test Macro F1:",
        result["best_metrics"]["macro_f1"]
    )

    print(
        "Saved model:",
        result["model_path"]
    )
