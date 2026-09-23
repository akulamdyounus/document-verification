import pytesseract
from PIL import Image

# Tesseract location on Windows
pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

image_path = "test_document.jpg"

image = Image.open(image_path)

text = pytesseract.image_to_string(image)

print("========== OCR RESULT ==========")
print(text)