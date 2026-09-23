import joblib
import pandas as pd

from pathlib import Path


# ============================================================
# MODEL PATH
# ============================================================

MODEL_PATH = Path(
    "models/authenticity_model.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

_model = None


def load_authenticity_model():

    global _model

    if _model is None:

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                "Authenticity model not found: "
                "models/authenticity_model.pkl"
            )

        _model = joblib.load(
            MODEL_PATH
        )

    return _model


# ============================================================
# PREDICT AUTHENTICITY
# ============================================================

def predict_authenticity(
    document_type,
    image_width,
    image_height,
    ocr_length,
    metadata_present,
    blur_score,
    compression_score
):

    model = load_authenticity_model()

    data = pd.DataFrame(
        [
            {
                "document_type": document_type,
                "image_width": image_width,
                "image_height": image_height,
                "ocr_length": ocr_length,
                "metadata_present": metadata_present,
                "blur_score": blur_score,
                "compression_score": compression_score
            }
        ]
    )

    prediction = model.predict(
        data
    )[0]

    probabilities = model.predict_proba(
        data
    )[0]

    classes = model.classes_

    probability_dict = {}

    for class_name, probability in zip(
        classes,
        probabilities
    ):

        probability_dict[
            int(class_name)
        ] = float(
            probability
        )

    suspicious_probability = (
        probability_dict.get(
            1,
            0.0
        )
        * 100
    )

    reference_probability = (
        probability_dict.get(
            0,
            0.0
        )
        * 100
    )

    if prediction == 1:

        result = "SUSPICIOUS"

    else:

        result = "REFERENCE / LOWER RISK"

    return (
        result,
        reference_probability,
        suspicious_probability
    )