
import numpy as np
import pandas as pd

from src.config import TARGET, ID_COLUMN


# ==================================================
# 1. DATASET CONFIGURATION
# ==================================================

EXPECTED_TARGETS = {
    "Beneficial",
    "Neutral",
    "Negative"
}

REQUIRED_FEATURES = {
    "Age",
    "Gender",
    "Academic_Level",
    "Primary_Platform",
    "Daily_Usage_Hours",
    "Weekend_Extra_Hours",
    "Device_Type",
    "Sleep_Duration_Hours",
    "Sleep_Quality_Score",
    "Late_Night_Usage",
    "Social_Comparison_Frequency",
    "Perceived_Stress_Score",
    "Mental_Health_Index",
    "Academic_Performance_GPA"
}

NUMERIC_COLUMNS = {
    "Age",
    "Daily_Usage_Hours",
    "Weekend_Extra_Hours",
    "Sleep_Duration_Hours",
    "Sleep_Quality_Score",
    "Perceived_Stress_Score",
    "Mental_Health_Index",
    "Academic_Performance_GPA"
}

CATEGORICAL_COLUMNS = {
    "Gender",
    "Academic_Level",
    "Primary_Platform",
    "Device_Type",
    "Social_Comparison_Frequency"
}

VALID_RANGES = {
    "Age": (10, 100),
    "Daily_Usage_Hours": (0, 24),
    "Weekend_Extra_Hours": (0, 24),
    "Sleep_Duration_Hours": (0, 24),
    "Sleep_Quality_Score": (1, 5),
    "Perceived_Stress_Score": (0, 40),
    "Mental_Health_Index": (0, 100),
    "Academic_Performance_GPA": (0, 4)
}

BOOLEAN_MAP = {
    "true": True,
    "false": False,
    "yes": True,
    "no": False,
    "1": True,
    "0": False
}

# Features permitted to enter the model.
ENGINEERED_FEATURES = {
    "Total_Usage_Hours",
    "Usage_to_Sleep_Ratio",
    "Late_Night_Usage_Hours_Flag"
}

ALLOWED_MODEL_FEATURES = (
    REQUIRED_FEATURES | ENGINEERED_FEATURES
)


# ==================================================
# 2. SCHEMA VALIDATION
# ==================================================

def validate_schema(data, allow_extra_columns=False):
    """
    Validate DataFrame type, empty data, column names,
    required columns and unexpected columns.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Input data must be a pandas DataFrame."
        )

    if data.empty:
        raise ValueError("The dataset is empty.")

    df = data.copy()

    # Normalize column names.
    df.columns = (
        df.columns.astype(str)
        .str.strip()
    )

    # Check empty column names.
    if any(not column for column in df.columns):
        raise ValueError(
            "Dataset contains empty column names."
        )

    # Check duplicate column names after normalization.
    if df.columns.duplicated().any():

        duplicates = (
            df.columns[df.columns.duplicated()]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Duplicate column names found: {duplicates}"
        )

    required_columns = (
        REQUIRED_FEATURES | {TARGET, ID_COLUMN}
    )

    actual_columns = set(df.columns)

    missing_columns = (
        required_columns - actual_columns
    )

    if missing_columns:
        raise ValueError(
            f"Required columns are missing: "
            f"{sorted(missing_columns)}"
        )

    unexpected_columns = (
        actual_columns - required_columns
    )

    if unexpected_columns and not allow_extra_columns:
        raise ValueError(
            f"Unexpected columns found: "
            f"{sorted(unexpected_columns)}"
        )

    return df


# ==================================================
# 3. LOAD DATASET
# ==================================================

def load_data(file_path, allow_extra_columns=False):
    """
    Load CSV and validate its schema.
    """

    try:
        data = pd.read_csv(file_path)

    except (
        FileNotFoundError,
        PermissionError,
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        UnicodeDecodeError,
        OSError
    ) as exc:

        raise ValueError(
            f"Unable to load CSV file: {exc}"
        ) from exc

    return validate_schema(
        data,
        allow_extra_columns=allow_extra_columns
    )


# ==================================================
# 4. DATA CLEANING AND VALIDATION
# ==================================================

def clean_data(
    data,
    allow_duplicate_ids=False,
    min_class_count=2,
    allow_extra_columns=False
):
    """
    Perform comprehensive data cleaning and validation.

    Invalid numeric and Boolean values become missing.
    Missing target rows are removed.
    Missing feature values are handled later by
    the training preprocessing pipeline.

    By default, each student must have one record.
    """

    if min_class_count < 2:
        raise ValueError(
            "min_class_count must be at least 2."
        )

    df = validate_schema(
        data,
        allow_extra_columns=allow_extra_columns
    )

    report = {
        "rows_before_cleaning": int(len(df)),
        "columns_before_cleaning": int(len(df)),
        "duplicate_rows_removed": 0,
        "missing_target_rows_removed": 0,
        "repeated_student_ids": 0,
        "numeric_conversion_errors": {},
        "infinite_values": {},
        "invalid_numeric_values": {},
        "invalid_boolean_values": 0,
        "missing_boolean_values": 0,
        "missing_values_by_column": {},
        "total_missing_values": 0,
        "constant_columns": [],
        "target_distribution": {}
    }

    # ------------------------------------------------
    # A. Remove exact duplicate records
    # ------------------------------------------------

    duplicate_count = int(df.duplicated().sum())

    df = df.drop_duplicates().copy()

    report["duplicate_rows_removed"] = duplicate_count

    if df.empty:
        raise ValueError(
            "Dataset contains no records after duplicate removal."
        )

    # ------------------------------------------------
    # B. Clean text values
    # ------------------------------------------------

    text_columns = df.select_dtypes(
        include=["object", "string"]
    ).columns

    for column in text_columns:

        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

        df[column] = df[column].replace(
            "", pd.NA
        )

    # ------------------------------------------------
    # C. Validate Student ID
    # ------------------------------------------------

    missing_id_count = int(
        df[ID_COLUMN].isna().sum()
    )

    if missing_id_count > 0:
        raise ValueError(
            f"{missing_id_count} rows contain missing "
            f"{ID_COLUMN} values."
        )

    # Validate unique IDs when one row per student.
    duplicate_mask = df[ID_COLUMN].duplicated(
        keep=False
    )

    duplicate_ids = (
        df.loc[duplicate_mask, ID_COLUMN]
        .unique()
        .tolist()
    )

    report["repeated_student_ids"] = len(
        duplicate_ids
    )

    if duplicate_ids and not allow_duplicate_ids:

        sample_ids = duplicate_ids[:10]

        raise ValueError(
            f"Duplicate student IDs found: {sample_ids}. "
            "Each student must have one record. "
            "If multiple records are intentional, enable "
            "allow_duplicate_ids and use group-based splitting."
        )

    # ------------------------------------------------
    # D. Validate target column
    # ------------------------------------------------

    df[TARGET] = (
        df[TARGET]
        .astype("string")
        .str.strip()
    )

    # Treat blank target values as missing.
    df[TARGET] = df[TARGET].replace(
        "", pd.NA
    )

    missing_target_count = int(
        df[TARGET].isna().sum()
    )

    df = df[df[TARGET].notna()].copy()

    report["missing_target_rows_removed"] = (
        missing_target_count
    )

    if df.empty:
        raise ValueError(
            "No records remain after removing missing target values."
        )

    # Normalize target labels without changing their meaning.
    target_mapping = {
        label.casefold(): label
        for label in EXPECTED_TARGETS
    }

    normalized_targets = (
        df[TARGET].str.casefold()
    )

    unexpected_targets = sorted(
        set(normalized_targets.unique())
        - set(target_mapping.keys())
    )

    if unexpected_targets:
        raise ValueError(
            f"Unexpected target labels: {unexpected_targets}. "
            f"Expected: {sorted(EXPECTED_TARGETS)}"
        )

    df[TARGET] = normalized_targets.map(
        target_mapping
    ).astype("string")

    # Validate number of target classes.
    class_counts = df[TARGET].value_counts()

    if len(class_counts) < 2:
        raise ValueError(
            "At least two target classes are required."
        )

    insufficient_classes = (
        class_counts[class_counts < min_class_count]
    )

    if not insufficient_classes.empty:
        raise ValueError(
            "Insufficient records in target classes: "
            f"{insufficient_classes.to_dict()}. "
            f"Minimum required per class: {min_class_count}"
        )

    # ------------------------------------------------
    # E. Validate numeric columns
    # ------------------------------------------------

    for column in NUMERIC_COLUMNS:

        original = df[column]

        converted = pd.to_numeric(
            original,
            errors="coerce"
        )

        # Count invalid text-to-number conversions.
        conversion_errors = int(
            (
                original.notna()
                & converted.isna()
            ).sum()
        )

        report["numeric_conversion_errors"][column] = (
            conversion_errors
        )

        # Detect infinite values.
        infinite_mask = np.isinf(
            converted.to_numpy(
                dtype=float,
                na_value=np.nan
            )
        )

        infinite_count = int(
            infinite_mask.sum()
        )

        report["infinite_values"][column] = (
            infinite_count
        )

        # Replace infinity with missing.
        converted = converted.replace(
            [np.inf, -np.inf],
            np.nan
        )

        df[column] = converted

    # ------------------------------------------------
    # F. Validate numeric ranges
    # ------------------------------------------------

    for column, (low, high) in VALID_RANGES.items():

        invalid_mask = (
            df[column].notna()
            & ~df[column].between(low, high)
        )

        invalid_count = int(
            invalid_mask.sum()
        )

        report["invalid_numeric_values"][column] = (
            invalid_count
        )

        # Do not clip invalid values.
        # Convert them to missing for later imputation.
        df.loc[invalid_mask, column] = np.nan

    # ------------------------------------------------
    # G. Validate Boolean column
    # ------------------------------------------------

    boolean_values = (
        df["Late_Night_Usage"]
        .astype("string")
        .str.strip()
        .str.lower()
    )

    original_missing = boolean_values.isna()

    converted_boolean = boolean_values.map(
        BOOLEAN_MAP
    )

    invalid_boolean_count = int(
        (
            ~original_missing
            & converted_boolean.isna()
        ).sum()
    )

    missing_boolean_count = int(
        original_missing.sum()
    )

    df["Late_Night_Usage"] = (
        converted_boolean.astype("boolean")
    )

    report["invalid_boolean_values"] = (
        invalid_boolean_count
    )

    report["missing_boolean_values"] = (
        missing_boolean_count
    )

    # ------------------------------------------------
    # H. Validate categorical columns
    # ------------------------------------------------

    categorical_report = {}

    for column in CATEGORICAL_COLUMNS:

        # Ensure consistent string representation.
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

        df[column] = df[column].replace(
            "", pd.NA
        )

        # Report available categories.
        categorical_report[column] = {
            str(value): int(count)
            for value, count in (
                df[column].value_counts().items()
            )
        }

    report["categorical_distribution"] = (
        categorical_report
    )

    # ------------------------------------------------
    # I. Check entirely missing feature columns
    # ------------------------------------------------

    empty_features = [
        column
        for column in REQUIRED_FEATURES
        if df[column].isna().all()
    ]

    if empty_features:
        raise ValueError(
            "The following feature columns contain "
            f"only missing values: {sorted(empty_features)}"
        )

    # ------------------------------------------------
    # J. Missing-value report
    # ------------------------------------------------

    missing_values = df.isna().sum()

    report["missing_values_by_column"] = {
        column: int(count)
        for column, count in missing_values.items()
        if count > 0
    }

    report["total_missing_values"] = int(
        missing_values.sum()
    )

    # ------------------------------------------------
    # K. Constant feature detection
    # ------------------------------------------------

    report["constant_columns"] = [
        column
        for column in REQUIRED_FEATURES
        if df[column].nunique(dropna=True) <= 1
    ]

    # Constant columns are reported, not automatically removed.
    # They may need further investigation.

    # ------------------------------------------------
    # L. Final target distribution
    # ------------------------------------------------

    report["target_distribution"] = {
        str(label): int(count)
        for label, count in (
            df[TARGET].value_counts().items()
        )
    }

    # ------------------------------------------------
    # M. Final dataset summary
    # ------------------------------------------------

    report["rows_after_cleaning"] = int(len(df))

    report["columns_after_cleaning"] = int(
        len(df.columns)
    )

    report["rows_removed"] = int(
        report["rows_before_cleaning"]
        - report["rows_after_cleaning"]
    )

    report["missing_values_after_cleaning"] = (
        report["total_missing_values"]
    )

    return df.reset_index(drop=True), report


# ==================================================
# 5. FEATURE ENGINEERING
# ==================================================

def engineer_features(data):
    """
    Generate row-level features.

    These transformations do not learn statistics
    from the complete dataset, so they can be applied
    before splitting when their source values are
    available at prediction time.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Feature engineering requires a DataFrame."
        )

    df = data.copy()

    original_target = (
        df[TARGET].copy()
        if TARGET in df.columns
        else None
    )

    # ------------------------------------------------
    # A. Total usage hours
    # ------------------------------------------------

    if {
        "Daily_Usage_Hours",
        "Weekend_Extra_Hours"
    }.issubset(df.columns):

        df["Total_Usage_Hours"] = (
            df["Daily_Usage_Hours"]
            + df["Weekend_Extra_Hours"]
        )

    # ------------------------------------------------
    # B. Usage-to-sleep ratio
    # ------------------------------------------------

    if {
        "Daily_Usage_Hours",
        "Sleep_Duration_Hours"
    }.issubset(df.columns):

        sleep_hours = (
            df["Sleep_Duration_Hours"]
            .replace(0, np.nan)
        )

        df["Usage_to_Sleep_Ratio"] = (
            df["Daily_Usage_Hours"]
            / sleep_hours
        )

    # ------------------------------------------------
    # C. Late-night usage hours
    # ------------------------------------------------

    if {
        "Daily_Usage_Hours",
        "Late_Night_Usage"
    }.issubset(df.columns):

        late_night = (
            df["Late_Night_Usage"]
            .astype("Float64")
        )

        df["Late_Night_Usage_Hours_Flag"] = (
            df["Daily_Usage_Hours"]
            * late_night
        )

    # ------------------------------------------------
    # D. Validate engineered features
    # ------------------------------------------------

    for column in ENGINEERED_FEATURES:

        if column not in df.columns:
            continue

        df[column] = df[column].replace(
            [np.inf, -np.inf],
            np.nan
        )

    # Ensure target was not modified.
    if original_target is not None:

        if not df[TARGET].equals(original_target):
            raise ValueError(
                "Feature engineering modified the target column."
            )

    return df


# ==================================================
# 6. PREPARE MODEL FEATURES
# ==================================================

def prepare_features(data):
    """
    Select approved features and separate target.

    Student ID and target are never included in X.

    Unexpected columns are excluded even if they
    were allowed during schema validation.
    """

    if not isinstance(data, pd.DataFrame):
        raise TypeError(
            "Input data must be a pandas DataFrame."
        )

    if TARGET not in data.columns:
        raise ValueError(
            f"Target column '{TARGET}' is missing."
        )

    if ID_COLUMN not in data.columns:
        raise ValueError(
            f"ID column '{ID_COLUMN}' is missing."
        )

    if data[TARGET].isna().any():
        raise ValueError(
            "Target contains missing values. "
            "Clean the dataset before training."
        )

    # Select only approved model features.
    selected_features = sorted(
        ALLOWED_MODEL_FEATURES.intersection(
            data.columns
        )
    )

    missing_features = (
        REQUIRED_FEATURES - set(selected_features)
    )

    if missing_features:
        raise ValueError(
            f"Required model features are missing: "
            f"{sorted(missing_features)}"
        )

    X = data[selected_features].copy()

    y = data[TARGET].copy()

    # Final target and ID leakage protection.
    forbidden_columns = {
        TARGET,
        ID_COLUMN
    }

    leaked_columns = (
        forbidden_columns.intersection(X.columns)
    )

    if leaked_columns:
        raise ValueError(
            f"Target or ID leakage detected: "
            f"{sorted(leaked_columns)}"
        )

    # Validate feature/target alignment.
    if len(X) != len(y):
        raise ValueError(
            "Feature and target row counts do not match."
        )

    if X.empty:
        raise ValueError(
            "No model features are available."
        )

    return X, y
