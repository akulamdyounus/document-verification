import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_validate

from sklearn.pipeline import Pipeline

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression


# ============================================================
# 1. LOAD DATASET
# ============================================================

data = pd.read_csv(
    "document_type_ml_dataset.csv"
)

print("=" * 60)
print("DOCUMENT CLASSIFIER EVALUATION")
print("=" * 60)

print()

print(
    "Total samples:",
    len(data)
)


# ============================================================
# 2. SHOW CLASS DISTRIBUTION
# ============================================================

print()
print("CLASS DISTRIBUTION")
print("-" * 60)

print(
    data["label"].value_counts()
)


# ============================================================
# 3. INPUT AND TARGET
# ============================================================

X = data["text"]

y = data["label"]


# ============================================================
# 4. CREATE ML PIPELINE
# ============================================================

pipeline = Pipeline(
    [
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1
            )
        ),

        (
            "classifier",
            LogisticRegression(
                max_iter=1000
            )
        )
    ]
)


# ============================================================
# 5. 5-FOLD CROSS VALIDATION
# ============================================================

cross_validator = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)


print()
print("Running 5-fold cross validation...")
print()


scores = cross_validate(
    pipeline,
    X,
    y,
    cv=cross_validator,
    scoring=[
        "accuracy",
        "precision_macro",
        "recall_macro",
        "f1_macro"
    ]
)


# ============================================================
# 6. CALCULATE RESULTS
# ============================================================

accuracy = scores[
    "test_accuracy"
]

precision = scores[
    "test_precision_macro"
]

recall = scores[
    "test_recall_macro"
]

f1 = scores[
    "test_f1_macro"
]


# ============================================================
# 7. DISPLAY FOLD RESULTS
# ============================================================

print("=" * 60)
print("FOLD RESULTS")
print("=" * 60)

for i in range(5):

    print(
        f"Fold {i + 1}:"
    )

    print(
        f"  Accuracy : "
        f"{accuracy[i] * 100:.2f}%"
    )

    print(
        f"  Precision: "
        f"{precision[i] * 100:.2f}%"
    )

    print(
        f"  Recall   : "
        f"{recall[i] * 100:.2f}%"
    )

    print(
        f"  F1 Score : "
        f"{f1[i] * 100:.2f}%"
    )

    print()


# ============================================================
# 8. AVERAGE RESULTS
# ============================================================

print("=" * 60)
print("AVERAGE RESULTS")
print("=" * 60)

print(
    f"Accuracy : "
    f"{accuracy.mean() * 100:.2f}%"
)

print(
    f"Precision: "
    f"{precision.mean() * 100:.2f}%"
)

print(
    f"Recall   : "
    f"{recall.mean() * 100:.2f}%"
)

print(
    f"F1 Score : "
    f"{f1.mean() * 100:.2f}%"
)


# ============================================================
# 9. STANDARD DEVIATION
# ============================================================

print()
print("=" * 60)
print("RESULT STABILITY")
print("=" * 60)

print(
    f"Accuracy standard deviation: "
    f"{accuracy.std() * 100:.2f}%"
)

print(
    f"F1 standard deviation: "
    f"{f1.std() * 100:.2f}%"
)


# ============================================================
# 10. FINAL WARNING
# ============================================================

print()
print("=" * 60)
print("IMPORTANT")
print("=" * 60)

print(
    "This evaluation uses SYNTHETIC training data."
)

print(
    "The scores do NOT represent real-world "
    "document classification accuracy."
)

print(
    "A production system needs a larger, "
    "diverse and properly sourced dataset."
)