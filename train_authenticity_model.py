import pandas as pd
import joblib

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# ============================================================
# FILE PATHS
# ============================================================

DATASET_PATH = "authenticity_dataset.csv"

MODEL_DIR = Path("models")

MODEL_PATH = MODEL_DIR / "authenticity_model.pkl"


# ============================================================
# LOAD DATASET
# ============================================================

print("=" * 60)
print("AUTHENTICITY MODEL TRAINING")
print("=" * 60)

print()

print("Loading dataset...")

df = pd.read_csv(DATASET_PATH)

print(
    f"Total samples: {len(df)}"
)

print()

print("Dataset columns:")

print(
    list(df.columns)
)

print()

print("Label distribution:")

print(
    df["label"].value_counts()
)


# ============================================================
# FEATURES
# ============================================================

feature_columns = [
    "document_type",
    "image_width",
    "image_height",
    "ocr_length",
    "metadata_present",
    "blur_score",
    "compression_score"
]


X = df[feature_columns]

y = df["label"]


# ============================================================
# CATEGORICAL FEATURES
# ============================================================

categorical_features = [
    "document_type"
]


# ============================================================
# NUMERICAL FEATURES
# ============================================================

numerical_features = [
    "image_width",
    "image_height",
    "ocr_length",
    "metadata_present",
    "blur_score",
    "compression_score"
]


# ============================================================
# PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer(
    transformers=[

        (
            "categorical",

            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),

                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]
            ),

            categorical_features
        ),

        (
            "numerical",

            Pipeline(
                steps=[
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),

                    (
                        "scaler",
                        StandardScaler()
                    )
                ]
            ),

            numerical_features
        )
    ]
)


# ============================================================
# MACHINE LEARNING MODEL
# ============================================================

classifier = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# ============================================================
# COMPLETE PIPELINE
# ============================================================

model = Pipeline(
    steps=[

        (
            "preprocessor",
            preprocessor
        ),

        (
            "classifier",
            classifier
        )
    ]
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print()

print(
    f"Training samples: {len(X_train)}"
)

print(
    f"Testing samples: {len(X_test)}"
)


# ============================================================
# TRAIN
# ============================================================

print()

print("Training authenticity model...")

model.fit(
    X_train,
    y_train
)

print(
    "Training completed."
)


# ============================================================
# TEST MODEL
# ============================================================

predictions = model.predict(
    X_test
)


accuracy = accuracy_score(
    y_test,
    predictions
)


print()

print("=" * 60)
print("MODEL ACCURACY")
print("=" * 60)

print(
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()

print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        predictions,
        target_names=[
            "Reference",
            "Suspicious"
        ],
        zero_division=0
    )
)


# ============================================================
# SAVE MODEL
# ============================================================

MODEL_DIR.mkdir(
    exist_ok=True
)

joblib.dump(
    model,
    MODEL_PATH
)


print()

print("=" * 60)
print("MODEL SAVED")
print("=" * 60)

print(
    MODEL_PATH
)

print()

print(
    "Authenticity model training finished."
)