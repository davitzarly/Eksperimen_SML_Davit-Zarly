from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "is_canceled"
DROP_COLUMNS = ["reservation_status", "reservation_status_date"]
RANDOM_STATE = 42


@dataclass(frozen=True)
class PreprocessingResult:
    train_path: Path
    test_path: Path
    metadata_path: Path
    preprocessor_path: Path
    feature_names_path: Path


def load_hotel_bookings(path: str | Path) -> pd.DataFrame:
    dataset_path = Path(path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset tidak ditemukan: {dataset_path.resolve()}")
    return pd.read_csv(dataset_path, na_values=["NULL", "NA", ""])


def clean_hotel_bookings(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned = cleaned.drop(columns=[col for col in DROP_COLUMNS if col in cleaned.columns], errors="ignore")

    if "children" in cleaned.columns:
        cleaned["children"] = cleaned["children"].fillna(0)
    if "agent" in cleaned.columns:
        cleaned["agent"] = cleaned["agent"].fillna(0)
    if "company" in cleaned.columns:
        cleaned["company"] = cleaned["company"].fillna(0)
    if "country" in cleaned.columns:
        cleaned["country"] = cleaned["country"].fillna("Unknown")

    if {"stays_in_weekend_nights", "stays_in_week_nights"}.issubset(cleaned.columns):
        cleaned["total_stay_nights"] = cleaned["stays_in_weekend_nights"] + cleaned["stays_in_week_nights"]
    if {"adults", "children", "babies"}.issubset(cleaned.columns):
        cleaned["total_guests"] = cleaned["adults"] + cleaned["children"] + cleaned["babies"]
    if {"reserved_room_type", "assigned_room_type"}.issubset(cleaned.columns):
        cleaned["room_changed"] = (cleaned["reserved_room_type"] != cleaned["assigned_room_type"]).astype(int)
    if {"total_of_special_requests", "required_car_parking_spaces"}.issubset(cleaned.columns):
        cleaned["has_special_need"] = (
            (cleaned["total_of_special_requests"] > 0) | (cleaned["required_car_parking_spaces"] > 0)
        ).astype(int)

    if "adr" in cleaned.columns:
        cleaned = cleaned[cleaned["adr"].between(0, 1000)]
    if "total_guests" in cleaned.columns:
        cleaned = cleaned[cleaned["total_guests"] > 0]

    if TARGET_COLUMN in cleaned.columns:
        cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].astype(int)
    return cleaned.reset_index(drop=True)


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Kolom target `{TARGET_COLUMN}` tidak ditemukan.")
    return df.drop(columns=[TARGET_COLUMN]), df[TARGET_COLUMN]


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    numeric_columns = df.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_columns = [col for col in df.columns if col not in numeric_columns]
    return numeric_columns, categorical_columns


def build_preprocessor(numeric_columns: Iterable[str], categorical_columns: Iterable[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, list(numeric_columns)),
            ("cat", categorical_pipeline, list(categorical_columns)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def run_preprocessing(
    input_path: str | Path,
    output_dir: str | Path,
    test_size: float = 0.2,
    random_state: int = RANDOM_STATE,
) -> PreprocessingResult:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    raw_df = load_hotel_bookings(input_path)
    cleaned_df = clean_hotel_bookings(raw_df)
    x, y = split_features_target(cleaned_df)
    numeric_columns, categorical_columns = get_feature_columns(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    preprocessor = build_preprocessor(numeric_columns, categorical_columns)
    x_train_processed = preprocessor.fit_transform(x_train)
    x_test_processed = preprocessor.transform(x_test)
    feature_names = list(preprocessor.get_feature_names_out())

    train_processed = pd.DataFrame(x_train_processed, columns=feature_names)
    train_processed[TARGET_COLUMN] = y_train.reset_index(drop=True)
    test_processed = pd.DataFrame(x_test_processed, columns=feature_names)
    test_processed[TARGET_COLUMN] = y_test.reset_index(drop=True)

    train_path = output_path / "train_preprocessed.csv"
    test_path = output_path / "test_preprocessed.csv"
    metadata_path = output_path / "metadata.json"
    preprocessor_path = output_path / "preprocessor.joblib"
    feature_names_path = output_path / "feature_names.txt"

    train_processed.to_csv(train_path, index=False, float_format="%.6g")
    test_processed.to_csv(test_path, index=False, float_format="%.6g")
    joblib.dump(preprocessor, preprocessor_path)
    feature_names_path.write_text("\n".join(feature_names), encoding="utf-8")
    metadata_path.write_text(
        pd.Series(
            {
                "raw_rows": len(raw_df),
                "clean_rows": len(cleaned_df),
                "train_rows": len(train_processed),
                "test_rows": len(test_processed),
                "target_column": TARGET_COLUMN,
                "numeric_columns": ",".join(numeric_columns),
                "categorical_columns": ",".join(categorical_columns),
                "random_state": random_state,
                "test_size": test_size,
            }
        ).to_json(indent=2),
        encoding="utf-8",
    )

    return PreprocessingResult(
        train_path=train_path,
        test_path=test_path,
        metadata_path=metadata_path,
        preprocessor_path=preprocessor_path,
        feature_names_path=feature_names_path,
    )
