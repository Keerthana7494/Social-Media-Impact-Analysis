
import json
import uuid
from datetime import datetime
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from src.config import (
    DATA_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    MODELS_DIR,
    TARGET,
    ID_COLUMN
)

from src.train import run_training

from src.data import (
    load_data,
    clean_data,
    engineer_features
)


# ============================================================
# PREDICTION HISTORY FILE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent
PREDICTIONS_DIR = PROJECT_ROOT / "predictions"
PREDICTION_HISTORY_FILE = PREDICTIONS_DIR / "prediction_history.csv"


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Social Media Impact MLOps",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM CSS - COLORFUL DASHBOARD
# ============================================================

st.markdown("""
<style>

/* Main application background */
.stApp {
    background-color: #F5F7FB;
}

/* Main title */
.main-title {
    font-size: 32px;
    font-weight: 800;
    color: #263B70;
    margin-bottom: 5px;
}

/* Subtitle */
.subtitle {
    color: #64748B;
    font-size: 15px;
    margin-bottom: 25px;
}

/* Section heading */
.section-title {
    color: #263B70;
    font-size: 22px;
    font-weight: 700;
    margin-top: 15px;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background: linear-gradient(135deg, #FFFFFF, #EEF2FF);
    border: 1px solid #DCE4F5;
    padding: 18px;
    border-radius: 12px;
    box-shadow: 0 3px 10px rgba(38, 59, 112, 0.06);
}

div[data-testid="stMetricLabel"] {
    color: #52617A;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #263B70;
    font-weight: 800;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(90deg, #4355DB, #7866E8);
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
    padding: 10px 20px;
}

.stButton > button:hover {
    background: linear-gradient(90deg, #3446C5, #6652D5);
    color: white;
    border: none;
}

/* Form submit button */
.stFormSubmitButton > button {
    background: linear-gradient(90deg, #0F9D8A, #16B89A);
    color: white;
    border-radius: 8px;
    font-weight: 700;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #EAF0FC;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    background-color: #E8EDFA;
    border-radius: 8px 8px 0 0;
    padding: 10px 18px;
    color: #263B70;
    font-weight: 600;
}

.stTabs [aria-selected="true"] {
    background-color: #4355DB;
    color: white;
}

/* Info boxes */
div[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Divider */
hr {
    border-color: #DCE4F5;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# 3. APPLICATION HEADER
# ============================================================

st.markdown(
    '<div class="main-title"> Social Media Impact on Student Life</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'End-to-end Machine Learning workflow with Streamlit, '
    'Scikit-learn and MLflow'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# 4. SIDEBAR
# ============================================================

default_data = DATA_DIR / "Social_media_impact_on_life.csv"

with st.sidebar:

    uploaded_file = st.file_uploader(
        "Upload a CSV dataset",
        type=["csv"]
    )

    # st.info(f"Target column: {TARGET}")

    # st.markdown("###  ML Workflow")

    # st.markdown("""
    # 1.  Data Loading
    # 2.  Data Cleaning
    # 3.  Exploratory Analysis
    # 4.  Feature Engineering
    # 5.  Model Comparison
    # 6.  Best Model Selection
    # 7.  Model Evaluation
    # 8.  MLflow Tracking
    # 9.  Model Versioning
    # """)

    # st.markdown("---")

    # st.caption("Social Media Impact Classification Project")


# ============================================================
# 5. LOAD DATASET
# ============================================================

try:

    if uploaded_file is not None:

        input_df = pd.read_csv(uploaded_file)

        # Save uploaded dataset for training
        active_path = DATA_DIR / "uploaded_dataset.csv"

        DATA_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        input_df.to_csv(
            active_path,
            index=False
        )

        st.sidebar.success("Uploaded dataset loaded.")

    else:

        active_path = default_data

        if default_data.exists():

            input_df = pd.read_csv(default_data)

        else:

            input_df = None

except Exception as error:

    st.error(f"Dataset loading failed: {error}")

    st.stop()


if input_df is None:

    st.warning(
        "Place the dataset inside the data folder "
        "or upload a CSV file."
    )

    st.stop()


# ============================================================
# 6. VALIDATE REQUIRED TARGET COLUMN
# ============================================================

if TARGET not in input_df.columns:

    st.error(
        f"Required target column '{TARGET}' was not found."
    )

    st.stop()


# ============================================================
# 7. CREATE TABS
# ============================================================

tab1, tab2, tab3, tab4 = st.tabs([

    " Data Overview",

    " Cleaning & EDA",

    " Train Models",

    " Prediction"

])


# ============================================================
# TAB 1: DATA OVERVIEW
# ============================================================

with tab1:

    st.markdown(
        '<div class="section-title">Dataset Overview</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Understand the structure and quality of the uploaded dataset."
    )

    # Dataset metrics
    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Records",
        f"{len(input_df):,}"
    )

    c2.metric(
        "Total Features",
        len(input_df.columns)
    )

    c3.metric(
        "Duplicate Rows",
        int(input_df.duplicated().sum())
    )

    c4.metric(
        "Missing Values",
        int(input_df.isna().sum().sum())
    )

    st.divider()

    # Dataset preview
    st.subheader("Dataset Preview")

    st.dataframe(
        input_df.head(20),
        width="stretch",
        hide_index=True
    )

    st.divider()

    # Data types and missing values
    st.subheader("Data Types and Missing Values")

    overview = pd.DataFrame({

        "Data Type": input_df.dtypes.astype(str),

        "Missing Count": input_df.isna().sum(),

        "Missing Percentage": (
            input_df.isna().mean() * 100
        ).round(2)

    })

    st.dataframe(
        overview.style.background_gradient(
            subset=["Missing Percentage"],
            cmap="YlOrRd"
        ),
        width="stretch"
    )

    st.divider()

    # Target distribution
    st.subheader("Target Class Distribution")

    target_counts = input_df[TARGET].value_counts()

    c1, c2 = st.columns([2, 1])

    with c1:

        st.bar_chart(
            target_counts,
            color="#635BDB"
        )

    with c2:

        st.dataframe(
            target_counts.rename("Records"),
            width="stretch"
        )


# ============================================================
# TAB 2: DATA CLEANING AND EDA
# ============================================================

with tab2:

    st.markdown(
        '<div class="section-title">Data Cleaning & Analysis</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Review the dataset before and after basic cleaning "
        "and inspect the feature engineering output."
    )

    try:

        # Basic cleaning
        cleaned, cleaning_report = clean_data(input_df)

        # Feature engineering
        featured = engineer_features(cleaned)

        feature_columns = [

            column

            for column in featured.columns

            if column not in [TARGET, ID_COLUMN]

        ]

        # --------------------------------------------
        # BEFORE CLEANING
        # --------------------------------------------

        st.subheader("1. Before Cleaning")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Original Rows",
            len(input_df)
        )

        c2.metric(
            "Original Columns",
            len(input_df.columns)
        )

        c3.metric(
            "Original Missing Values",
            int(input_df.isna().sum().sum())
        )

        st.subheader("Missing Values Before Imputation")

        missing_before = featured[feature_columns].isna().sum()

        missing_before = missing_before[
            missing_before > 0
        ]

        if not missing_before.empty:

            st.dataframe(
                missing_before.rename("Missing Values"),
                width="stretch"
            )

        else:

            st.success("No missing feature values found.")

        st.metric(
            "Total Missing Values Before Imputation",
            int(
                featured[feature_columns].isna().sum().sum()
            )
        )

        st.divider()

        # --------------------------------------------
        # AFTER CLEANING
        # --------------------------------------------

        st.subheader("2. After Cleaning")

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Cleaned Rows",
            len(cleaned)
        )

        c2.metric(
            "Cleaned Columns",
            len(cleaned.columns)
        )

        c3.metric(
            "Missing Values After Basic Cleaning",
            int(cleaned.isna().sum().sum())
        )

        #st.subheader("Cleaning Report")

        #st.json(cleaning_report)

        st.divider()

        # --------------------------------------------
        # IMPUTATION REPORT
        # --------------------------------------------

        st.subheader("3. Missing Values After Imputation")

        cleaned_file = (
            REPORTS_DIR / "cleaned_dataset.csv"
        )

        if cleaned_file.exists():

            imputed_data = pd.read_csv(cleaned_file)

            available_columns = [

                col

                for col in feature_columns

                if col in imputed_data.columns

            ]

            missing_after = (
                imputed_data[available_columns].isna().sum()
            )

            c1, c2 = st.columns(2)

            c1.metric(
                "Missing Before Imputation",
                int(
                    featured[feature_columns].isna().sum().sum()
                )
            )

            c2.metric(
                "Missing After Imputation",
                int(missing_after.sum())
            )

            st.dataframe(
                missing_after.rename("Missing Values"),
                width="stretch"
            )

            if missing_after.sum() == 0:

                st.success(
                    "All feature missing values handled in the "
                    "generated cleaned dataset report."
                )

            else:

                st.warning(
                    "Some missing values remain. "
                    "Review the cleaning report."
                )

        else:

            st.info(
                "Run the training pipeline to generate "
                "the imputed dataset report."
            )

        st.divider()

        # --------------------------------------------
        # FEATURE ENGINEERING
        # --------------------------------------------

        st.subheader("4. Feature Engineering Preview")

        st.dataframe(
            featured.head(10),
            width="stretch",
            hide_index=True
        )

        st.caption(
            "Engineered features include total usage hours "
            "and usage-to-sleep ratio, if defined in data.py."
        )

        st.divider()

        # --------------------------------------------
        # SAVED EDA FIGURES
        # --------------------------------------------

        st.subheader("5. Exploratory Data Analysis Visualizations")

        figures = [

            (
                "numeric_feature_distributions.png",
                "Feature Distributions"
            ),

            (
                "correlation_heatmap.png",
                "Correlation Heatmap"
            ),

            (
                "target_distribution.png",
                "Target Distribution"
            )

        ]

        for filename, title in figures:

            image_path = FIGURES_DIR / filename

            if image_path.exists():

                st.markdown(f"**{title}**")

                st.image(
                    str(image_path),
                    width="stretch"
                )

    except Exception as error:

        st.error(
            f"Data analysis failed: {error}"
        )


# ============================================================
# TAB 3: MODEL TRAINING AND COMPARISON
# ============================================================

with tab3:

    st.markdown(
        '<div class="section-title">Machine Learning Model Training</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Train and compare five classification algorithms "
        "using the same training and validation datasets."
    )

    # --------------------------------------------
    # MODELS INFORMATION
    # --------------------------------------------

    model_names = [

        "Dummy Classifier",

        "Logistic Regression",

        "Decision Tree",

        "Random Forest",

        "Gradient Boosting"

    ]

    st.subheader("Models Included")

    cols = st.columns(5)

    for column, name in zip(cols, model_names):

        with column:

            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg,#E8EDFF,#FFFFFF);
                    border: 1px solid #CBD5F5;
                    padding: 14px;
                    border-radius: 10px;
                    text-align: center;
                    min-height: 85px;
                    color: #263B70;
                    font-weight: 700;
                ">
                    {name}
                </div>
                """,
                unsafe_allow_html=True
            )

    st.divider()

    st.info(
        "Primary selection metric: Macro F1. "
        "Balanced Accuracy is used as the tie-breaker."
    )

    # --------------------------------------------
    # RUN TRAINING
    # --------------------------------------------

    if st.button(
        "Run Complete Training Pipeline",
        type="primary",
        use_container_width=True
    ):

        with st.spinner(
            "Cleaning data, training five models, "
            "comparing results and tracking experiments..."
        ):

            try:

                result = run_training(active_path)

                st.session_state["training_result"] = result

                st.success(
                    f"Training completed! Best model: "
                    f"{result['best_model']}"
                )

            except Exception as error:

                st.error(
                    f"Training failed: {error}"
                )

    # --------------------------------------------
    # DISPLAY TRAINING RESULTS
    # --------------------------------------------

    result = st.session_state.get(
        "training_result"
    )

    if result:

        st.divider()

        st.subheader("Model Comparison Results")

        comparison = result["comparison"]

        st.dataframe(
            comparison.style.background_gradient(
                subset=[
                    "accuracy",
                    "balanced_accuracy",
                    "macro_f1",
                    "weighted_f1"
                ],
                cmap="YlGn"
            ).format({
                "accuracy": "{:.4f}",
                "balanced_accuracy": "{:.4f}",
                "macro_f1": "{:.4f}",
                "weighted_f1": "{:.4f}"
            }),
            width="stretch",
            hide_index=True
        )

        # ----------------------------------------
        # MODEL COMPARISON CHART
        # ----------------------------------------

        st.subheader("Model Comparison Visualization")

        chart_path = (
            FIGURES_DIR / "model_comparison.png"
        )

        if chart_path.exists():

            st.image(
                str(chart_path),
                width="stretch"
            )

        # ----------------------------------------
        # HYPERPARAMETER TUNING RESULTS
        # ----------------------------------------

        st.divider()
        st.subheader("Hyperparameter Tuning Results")
        st.write(
            "GridSearchCV tunes only the model selected from the original "
            "validation comparison. Cross-validation uses the training split; "
            "the test split remains untouched until final evaluation."
        )

        tuned_metrics = result.get("tuned_validation_metrics", {})
        best_params = result.get("best_hyperparameters", {})

        if best_params:
            st.markdown("**Best Hyperparameters**")
            st.json(best_params)

        st.metric(
            "Best Cross-Validation Macro F1",
            f"{result.get('cross_validation_macro_f1', 0):.4f}"
        )

        if tuned_metrics:
            st.markdown("**Original vs Tuned Validation Metrics**")
            before_after = result.get("before_after_tuning")
            if before_after is not None:
                st.dataframe(
                    before_after.style.format({
                        "accuracy": "{:.4f}",
                        "balanced_accuracy": "{:.4f}",
                        "macro_f1": "{:.4f}",
                        "weighted_f1": "{:.4f}"
                    }),
                    width="stretch",
                    hide_index=True
                )

        tuning_figures = [
            ("hyperparameter_tuning_scores.png", "Hyperparameter Search CV Scores"),
            ("validation_before_after_tuning.png", "Validation Before vs After Tuning"),
            ("final_test_metrics.png", "Final Test Metrics"),
        ]
        for filename, title in tuning_figures:
            figure_path = FIGURES_DIR / filename
            if figure_path.exists():
                st.markdown(f"**{title}**")
                st.image(str(figure_path), width="stretch")

        tuning_csv_path = REPORTS_DIR / "hyperparameter_tuning_results.csv"
        if tuning_csv_path.exists():
            with open(tuning_csv_path, "rb") as tuning_file:
                st.download_button(
                    "Download Hyperparameter Tuning Results (CSV)",
                    data=tuning_file.read(),
                    file_name="hyperparameter_tuning_results.csv",
                    mime="text/csv"
                )

        params_path = REPORTS_DIR / "best_hyperparameters.json"
        if params_path.exists():
            with open(params_path, "rb") as params_file:
                st.download_button(
                    "Download Best Hyperparameters (JSON)",
                    data=params_file.read(),
                    file_name="best_hyperparameters.json",
                    mime="application/json"
                )

        # ----------------------------------------
        # BEST MODEL
        # ----------------------------------------

        st.divider()

        st.subheader(" Selected Best Model")

        st.success(
            f"Best Model: {result['best_model']}"
        )

        validation_metrics = result.get(
            "validation_metrics",
            {}
        )

        test_metrics = result["best_metrics"]

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Test Accuracy",
            f"{test_metrics['accuracy']:.4f}"
        )

        c2.metric(
            "Test Macro F1",
            f"{test_metrics['macro_f1']:.4f}"
        )

        c3.metric(
            "Test Balanced Accuracy",
            f"{test_metrics['balanced_accuracy']:.4f}"
        )

        c1, c2 = st.columns(2)

        c1.metric(
            "Validation Macro F1",
            f"{validation_metrics.get('macro_f1', 0):.4f}"
        )

        c2.metric(
            "Test Weighted F1",
            f"{test_metrics['weighted_f1']:.4f}"
        )

        # ----------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------

        st.divider()

        st.subheader(" Final Model Confusion Matrix")

        confusion_path = (
            FIGURES_DIR / "confusion_matrix.png"
        )

        if confusion_path.exists():

            st.image(
                str(confusion_path),
                width="stretch"
            )

        # ----------------------------------------
        # CLASSIFICATION REPORT
        # ----------------------------------------

        report_path = (
            REPORTS_DIR / "classification_report.json"
        )

        if report_path.exists():

            with open(
                report_path,
                encoding="utf-8"
            ) as file:

                classification_data = json.load(file)

            st.subheader("Classification Report")

            report_df = pd.DataFrame(
                classification_data
            ).transpose()

            st.dataframe(
                report_df.style.format(
                    precision=4
                ).background_gradient(
                    cmap="Blues"
                ),
                width="stretch"
            )

        # ----------------------------------------
        # MODEL VERSION INFORMATION
        # ----------------------------------------

        st.divider()

        st.subheader(" Saved Model Information")

        c1, c2 = st.columns(2)

        c1.metric(
            "Model Version",
            f"v{result['model_version']}"
        )

        c2.metric(
            "Selected Model",
            result["best_model"]
        )

        st.write(
            "**Model file:**",
            result["model_path"]
        )

        st.write(
            "**Reports directory:**",
            str(REPORTS_DIR)
        )

        st.success(
            "Model, metadata and latest model reference saved successfully."
        )


# ============================================================
# TAB 4: MODEL PREDICTION
# ============================================================

with tab4:

    st.markdown(
        '<div class="section-title">Predict Social Media Impact</div>',
        unsafe_allow_html=True
    )

    st.write(
        "Enter student information to predict the overall "
        "impact of social media usage."
    )

    latest_file = (
        MODELS_DIR / "latest_model.json"
    )

    if not latest_file.exists():

        st.info(
            "Train the models first to enable predictions."
        )

    else:

        try:

            # ----------------------------------------
            # LOAD LATEST MODEL
            # ----------------------------------------

            with open(
                latest_file,
                encoding="utf-8"
            ) as file:

                latest = json.load(file)

            project_root = Path(__file__).resolve().parent

            model_path = (
                project_root / latest["model_path"]
            )

            metadata_path = (
                project_root / latest["metadata_path"]
            )

            if not model_path.exists():

                st.error(
                    f"Model file not found: {model_path}"
                )

                st.stop()

            model = joblib.load(model_path)

            # Load metadata
            if metadata_path.exists():

                with open(
                    metadata_path,
                    encoding="utf-8"
                ) as file:

                    metadata = json.load(file)

            else:

                metadata = {}

            st.success(
                f"Loaded Model: {metadata.get('model_name', 'Latest Model')} "
                f"| Version: v{metadata.get('model_version', 'Unknown')}"
            )

            # ----------------------------------------
            # PREDICTION FORM
            # ----------------------------------------

            with st.form("prediction_form"):

                st.subheader(" Student Information")

                a, b, c = st.columns(3)

                age = a.number_input(
                    "Age",
                    min_value=10,
                    max_value=100,
                    value=19
                )

                gender = b.selectbox(
                    "Gender",
                    [
                        "Female",
                        "Male",
                        "Non-Binary",
                        "Prefer not to say"
                    ]
                )

                level = c.selectbox(
                    "Academic Level",
                    [
                        "High School",
                        "Undergraduate",
                        "Postgraduate"
                    ]
                )

                st.divider()

                st.subheader(" Social Media Usage")

                a, b, c = st.columns(3)

                platform = a.selectbox(
                    "Primary Platform",
                    [
                        "Instagram",
                        "TikTok",
                        "YouTube",
                        "Snapchat",
                        "Reddit",
                        "X (Twitter)",
                        "LinkedIn"
                    ]
                )

                daily = b.number_input(
                    "Daily Usage Hours",
                    min_value=0.0,
                    max_value=24.0,
                    value=4.5
                )

                weekend = c.number_input(
                    "Weekend Extra Hours",
                    min_value=0.0,
                    max_value=24.0,
                    value=1.5
                )

                device = a.selectbox(
                    "Device Type",
                    [
                        "Smartphone",
                        "Laptop/PC",
                        "Tablet"
                    ]
                )

                late_night = b.selectbox(
                    "Late Night Usage",
                    [
                        True,
                        False
                    ]
                )

                comparison = c.selectbox(
                    "Social Comparison Frequency",
                    [
                        "Never",
                        "Rarely",
                        "Sometimes",
                        "Frequently",
                        "Always"
                    ]
                )

                st.divider()

                st.subheader(" Health and Academic Information")

                a, b, c = st.columns(3)

                sleep = a.number_input(
                    "Sleep Duration Hours",
                    min_value=0.1,
                    max_value=24.0,
                    value=7.0
                )

                sleep_quality = b.slider(
                    "Sleep Quality Score",
                    1,
                    5,
                    3
                )

                stress = c.number_input(
                    "Perceived Stress Score",
                    min_value=0.0,
                    max_value=40.0,
                    value=13.0
                )

                mental = a.slider(
                    "Mental Health Index",
                    0,
                    100,
                    80
                )

                gpa = b.number_input(
                    "Academic Performance GPA",
                    min_value=0.0,
                    max_value=4.0,
                    value=3.4
                )

                submitted = st.form_submit_button(
                    "Predict Social Media Impact",
                    use_container_width=True
                )

            # ----------------------------------------
            # MAKE PREDICTION
            # ----------------------------------------

            if submitted:

                # Keep the original form values for prediction history.
                raw_sample = pd.DataFrame([{
                    "Age": age,
                    "Gender": gender,
                    "Academic_Level": level,
                    "Primary_Platform": platform,
                    "Daily_Usage_Hours": daily,
                    "Weekend_Extra_Hours": weekend,
                    "Device_Type": device,
                    "Sleep_Duration_Hours": sleep,
                    "Sleep_Quality_Score": sleep_quality,
                    "Late_Night_Usage": late_night,
                    "Social_Comparison_Frequency": comparison,
                    "Perceived_Stress_Score": stress,
                    "Mental_Health_Index": mental,
                    "Academic_Performance_GPA": gpa
                }])

                # Create a separate copy for feature engineering/model input.
                sample = engineer_features(raw_sample.copy())

                # Match the exact training feature schema
                expected_features = metadata.get(
                    "feature_columns"
                )

                if expected_features:

                    missing_features = [

                        col

                        for col in expected_features

                        if col not in sample.columns

                    ]

                    if missing_features:

                        raise ValueError(
                            "Prediction input is missing required "
                            f"features: {missing_features}"
                        )

                    sample = sample[expected_features]

                # Predict class
                prediction = model.predict(sample)[0]

                st.divider()

                st.subheader(" Prediction Result")

                if prediction == "Beneficial":

                    st.success(
                        f"Predicted Overall Impact: {prediction} "
                    )

                elif prediction == "Negative":

                    st.error(
                        f"Predicted Overall Impact: {prediction} "
                    )

                else:

                    st.info(
                        f"Predicted Overall Impact: {prediction} "
                    )

                # ------------------------------------
                # CLASS PROBABILITIES
                # ------------------------------------

                if hasattr(model, "predict_proba"):

                    probabilities = model.predict_proba(
                        sample
                    )[0]

                    probability_df = pd.DataFrame({

                        "Class": model.classes_,

                        "Probability": probabilities

                    })

                    probability_df = probability_df.sort_values(
                        "Probability",
                        ascending=False
                    )

                    st.subheader("Class Prediction Probabilities")

                    st.bar_chart(
                        probability_df.set_index("Class"),
                        color="#16A085"
                    )

                    st.dataframe(
                        probability_df.style.format({
                            "Probability": "{:.2%}"
                        }).background_gradient(
                            subset=["Probability"],
                            cmap="Greens"
                        ),
                        width="stretch",
                        hide_index=True
                    )

                # ------------------------------------
                # SAVE PREDICTION TO A SEPARATE CSV
                # ------------------------------------

                prediction_record = raw_sample.copy()
                prediction_record.insert(
                    0,
                    "Prediction_ID",
                    str(uuid.uuid4())
                )
                prediction_record["Prediction_Date"] = (
                    datetime.now().astimezone().isoformat(timespec="seconds")
                )
                prediction_record["Predicted_Impact"] = str(prediction)
                prediction_record["Model_Name"] = metadata.get(
                    "model_name", "Unknown"
                )
                prediction_record["Model_Version"] = metadata.get(
                    "model_version", latest.get("model_version", "Unknown")
                )

                # Store each class probability in its own CSV column.
                if hasattr(model, "predict_proba"):
                    for class_name, probability in zip(
                        model.classes_, probabilities
                    ):
                        safe_class_name = str(class_name).strip().replace(" ", "_")
                        prediction_record[
                            f"Probability_{safe_class_name}"
                        ] = float(probability)

                try:
                    PREDICTIONS_DIR.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    prediction_record.to_csv(
                        PREDICTION_HISTORY_FILE,
                        mode="a",
                        header=not PREDICTION_HISTORY_FILE.exists(),
                        index=False
                    )

                    st.success(
                        "Prediction saved to predictions/prediction_history.csv"
                    )

                except (OSError, PermissionError) as save_error:
                    st.error(
                        f"Prediction was generated, but could not be saved: {save_error}"
                    )

        except Exception as error:

            st.error(
                f"Prediction failed: {error}"
            )

    # --------------------------------------------
    # PREDICTION HISTORY
    # --------------------------------------------

    st.divider()
    st.subheader("Prediction History")

    if PREDICTION_HISTORY_FILE.exists():
        try:
            history_df = pd.read_csv(PREDICTION_HISTORY_FILE)

            # history_col1, history_col2 = st.columns(2)
            # history_col1.metric("Total Predictions", len(history_df))

            # if "Predicted_Impact" in history_df.columns:
            #     history_col2.metric(
            #         "Distinct Predicted Classes",
            #         history_df["Predicted_Impact"].nunique()
            #     )

            # st.dataframe(
            #     history_df.sort_values(
            #         "Prediction_Date",
            #         ascending=False
            #     ) if "Prediction_Date" in history_df.columns else history_df,
            #     width="stretch",
            #     hide_index=True
            # )

            st.download_button(
                "Download Prediction History (CSV)",
                data=history_df.to_csv(index=False).encode("utf-8"),
                file_name="prediction_history.csv",
                mime="text/csv"
            )

        except Exception as history_error:
            st.error(
                f"Could not read prediction history: {history_error}"
            )
    else:
        st.info(
            "No prediction history yet. Submit the prediction form to save the first record."
        )


# ============================================================
# 8. FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        color:#64748B;
        padding:12px;
        font-size:13px;
    ">
        Social Media Impact on Student Life |
        Machine Learning & MLOps Project
    </div>
    """,
    unsafe_allow_html=True
)
