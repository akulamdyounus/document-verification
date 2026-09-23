import streamlit as st
import fitz
import pytesseract
import joblib

from PIL import Image
import io
from pathlib import Path


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Document Verifier",
    page_icon="📄",
    layout="centered"
)


# ============================================================
# TESSERACT CONFIGURATION
# ============================================================

TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# ============================================================
# ML MODEL PATH
# ============================================================

MODEL_PATH = Path(
    "models/document_type_model.pkl"
)


# ============================================================
# LOAD MACHINE LEARNING MODEL
# ============================================================

@st.cache_resource
def load_ml_model():

    if not MODEL_PATH.exists():

        return None, None

    model_data = joblib.load(
        MODEL_PATH
    )

    model = model_data["model"]

    vectorizer = model_data["vectorizer"]

    return model, vectorizer


ml_model, vectorizer = load_ml_model()


# ============================================================
# ML DOCUMENT TYPE PREDICTION
# ============================================================

def predict_document_type(text):

    if not text.strip():

        return (
            "Unknown",
            0.0,
            {}
        )


    # Convert OCR text into TF-IDF features

    text_vector = vectorizer.transform(
        [text]
    )


    # Predict document type

    prediction = ml_model.predict(
        text_vector
    )[0]


    # Get probabilities

    probabilities = ml_model.predict_proba(
        text_vector
    )[0]


    classes = ml_model.classes_


    probability_dict = {}

    for class_name, probability in zip(
        classes,
        probabilities
    ):

        probability_dict[class_name] = (
            probability * 100
        )


    confidence = max(
        probabilities
    ) * 100


    return (
        prediction,
        confidence,
        probability_dict
    )


# ============================================================
# WEBSITE HEADER
# ============================================================

st.title(
    "📄 AI Document Verifier"
)

st.write(
    "Upload a PDF or image to analyze the document."
)


st.info(
    "Current version: document classification using OCR + Machine Learning."
)


# ============================================================
# CHECK ML MODEL
# ============================================================

if ml_model is None:

    st.error(
        "Machine learning model was not found."
    )

    st.write(
        "Make sure this file exists:"
    )

    st.code(
        "models/document_type_model.pkl"
    )

    st.stop()


# ============================================================
# FILE UPLOADER
# ============================================================

uploaded_file = st.file_uploader(
    "Choose a document",
    type=[
        "pdf",
        "jpg",
        "jpeg",
        "png"
    ]
)


# ============================================================
# PROCESS FILE
# ============================================================

if uploaded_file is not None:

    st.success(
        "Document uploaded successfully!"
    )


    # ========================================================
    # BASIC FILE INFORMATION
    # ========================================================

    st.subheader(
        "📋 File Information"
    )

    st.write(
        "**File name:**",
        uploaded_file.name
    )

    st.write(
        "**File type:**",
        uploaded_file.type
    )

    st.write(
        "**File size:**",
        f"{uploaded_file.size:,} bytes"
    )


    # ========================================================
    # IMAGE PROCESSING
    # ========================================================

    if uploaded_file.type in [
        "image/jpeg",
        "image/png"
    ]:

        st.subheader(
            "🖼️ Image Analysis"
        )


        # Read image

        image_bytes = uploaded_file.read()

        image = Image.open(
            io.BytesIO(image_bytes)
        )


        # Display image

        st.image(
            image,
            caption="Uploaded Document",
            use_container_width=True
        )


        # ====================================================
        # OCR
        # ====================================================

        st.subheader(
            "🔎 OCR Analysis"
        )


        with st.spinner(
            "Reading document with OCR..."
        ):

            extracted_text = (
                pytesseract.image_to_string(
                    image
                )
            )


        # ====================================================
        # OCR RESULT
        # ====================================================

        if extracted_text.strip():

            st.success(
                "Text detected successfully!"
            )


            st.text_area(
                "Extracted Text",
                extracted_text,
                height=300
            )


            # =================================================
            # ML CLASSIFICATION
            # =================================================

            st.subheader(
                "🤖 ML Document Classification"
            )


            (
                document_type,
                confidence,
                probabilities
            ) = predict_document_type(
                extracted_text
            )


            st.info(
                f"Document Type: {document_type}"
            )


            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )


            # =================================================
            # PROBABILITIES
            # =================================================

            st.subheader(
                "📊 Classification Probabilities"
            )


            for class_name, probability in (
                probabilities.items()
            ):

                st.write(
                    f"**{class_name}:** "
                    f"{probability:.2f}%"
                )

                st.progress(
                    min(
                        int(probability),
                        100
                    )
                )


        else:

            st.warning(
                "No text could be detected "
                "in this image."
            )


    # ========================================================
    # PDF PROCESSING
    # ========================================================

    elif uploaded_file.type == "application/pdf":

        st.subheader(
            "📑 PDF Analysis"
        )


        # Read PDF

        pdf_bytes = uploaded_file.read()


        document = fitz.open(
            stream=pdf_bytes,
            filetype="pdf"
        )


        # ====================================================
        # PDF PAGE COUNT
        # ====================================================

        st.write(
            "**Number of pages:**",
            len(document)
        )


        # ====================================================
        # PDF METADATA
        # ====================================================

        st.subheader(
            "📋 PDF Metadata"
        )


        metadata = document.metadata


        st.write(
            "**Title:**",
            metadata.get(
                "title",
                "Not available"
            )
        )


        st.write(
            "**Author:**",
            metadata.get(
                "author",
                "Not available"
            )
        )


        st.write(
            "**Creator:**",
            metadata.get(
                "creator",
                "Not available"
            )
        )


        st.write(
            "**Producer:**",
            metadata.get(
                "producer",
                "Not available"
            )
        )


        st.write(
            "**Creation date:**",
            metadata.get(
                "creationDate",
                "Not available"
            )
        )


        st.write(
            "**Modification date:**",
            metadata.get(
                "modDate",
                "Not available"
            )
        )


        # ====================================================
        # EXTRACT TEXT FROM PDF
        # ====================================================

        pdf_text = ""


        for page in document:

            page_text = page.get_text()

            pdf_text += page_text + "\n"


        # ====================================================
        # IF PDF HAS TEXT
        # ====================================================

        if pdf_text.strip():

            st.subheader(
                "📝 PDF Text"
            )


            st.text_area(
                "Extracted PDF Text",
                pdf_text,
                height=300
            )


            # =================================================
            # ML CLASSIFICATION
            # =================================================

            st.subheader(
                "🤖 ML Document Classification"
            )


            (
                document_type,
                confidence,
                probabilities
            ) = predict_document_type(
                pdf_text
            )


            st.info(
                f"Document Type: {document_type}"
            )


            st.metric(
                "Model Confidence",
                f"{confidence:.2f}%"
            )


            st.subheader(
                "📊 Classification Probabilities"
            )


            for class_name, probability in (
                probabilities.items()
            ):

                st.write(
                    f"**{class_name}:** "
                    f"{probability:.2f}%"
                )


                st.progress(
                    min(
                        int(probability),
                        100
                    )
                )


        # ====================================================
        # SCANNED PDF
        # ====================================================

        else:

            st.warning(
                "This PDF does not contain "
                "selectable text."
            )


            st.info(
                "It may be a scanned PDF. "
                "Running OCR on the PDF pages..."
            )


            ocr_pdf_text = ""


            progress = st.progress(
                0
            )


            total_pages = len(document)


            for page_number, page in enumerate(
                document
            ):

                # Render PDF page as image

                pixmap = page.get_pixmap(
                    matrix=fitz.Matrix(
                        2,
                        2
                    )
                )


                image = Image.frombytes(
                    "RGB",
                    [
                        pixmap.width,
                        pixmap.height
                    ],
                    pixmap.samples
                )


                # OCR page

                page_text = (
                    pytesseract.image_to_string(
                        image
                    )
                )


                ocr_pdf_text += (
                    page_text + "\n"
                )


                progress.progress(
                    (page_number + 1)
                    / total_pages
                )


            # =================================================
            # DISPLAY OCR TEXT
            # =================================================

            if ocr_pdf_text.strip():

                st.success(
                    "OCR text detected successfully!"
                )


                st.text_area(
                    "OCR Extracted PDF Text",
                    ocr_pdf_text,
                    height=300
                )


                # =============================================
                # ML CLASSIFICATION
                # =============================================

                st.subheader(
                    "🤖 ML Document Classification"
                )


                (
                    document_type,
                    confidence,
                    probabilities
                ) = predict_document_type(
                    ocr_pdf_text
                )


                st.info(
                    f"Document Type: {document_type}"
                )


                st.metric(
                    "Model Confidence",
                    f"{confidence:.2f}%"
                )


                st.subheader(
                    "📊 Classification Probabilities"
                )


                for class_name, probability in (
                    probabilities.items()
                ):

                    st.write(
                        f"**{class_name}:** "
                        f"{probability:.2f}%"
                    )


                    st.progress(
                        min(
                            int(probability),
                            100
                        )
                    )


            else:

                st.error(
                    "OCR could not detect text "
                    "in this PDF."
                )


        # Close PDF

        document.close()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Document Verifier — "
    "Development version"
)