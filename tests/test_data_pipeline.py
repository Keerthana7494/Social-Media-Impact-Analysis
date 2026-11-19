
import pytest

from src.config import DATA_DIR
from src.data import (
    load_data,
    clean_data,
    engineer_features,
    prepare_features
)


@pytest.fixture(scope="module")
def processed_data():

    # 1. Load original dataset
    file_path = DATA_DIR / "Social_media_impact_on_life.csv"

    data = load_data(file_path)

    # 2. Clean dataset
    cleaned_data, report = clean_data(data)

    # 3. Generate new features
    featured_data = engineer_features(cleaned_data)

    return featured_data


def test_dataset_loading():

    file_path = DATA_DIR / "Social_media_impact_on_life.csv"

    data = load_data(file_path)

    assert not data.empty
    assert "Overall_Impact" in data.columns


def test_data_cleaning(processed_data):

    assert not processed_data.empty
    assert processed_data["Overall_Impact"].notna().all()


def test_feature_engineering(processed_data):

    assert "Total_Usage_Hours" in processed_data.columns
    assert "Usage_to_Sleep_Ratio" in processed_data.columns
    assert "Late_Night_Usage_Hours_Flag" in processed_data.columns


def test_feature_target_separation(processed_data):

    X, y = prepare_features(processed_data)

    assert len(X) == len(y)

    assert "Student_ID" not in X.columns
    assert "Overall_Impact" not in X.columns

    assert set(y.unique()) <= {
        "Beneficial",
        "Neutral",
        "Negative"
    }
