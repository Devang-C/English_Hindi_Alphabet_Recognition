from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model


app = FastAPI()

# Load the saved model
model_path = 'models/improved_alphabet_recognition_model_2.h5'  # Replace with the actual path to your saved model
model = load_model(model_path)

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def read_item(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


def manual_mapping(class_index):
    manual_mapping_dict = {
        0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
        10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
        20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'ञ',
        27: 'ट', 28: 'ठ', 29: 'ड',
        30: 'ढ', 31: 'ण', 32: 'त',
        33: 'थ', 34: 'द', 35: 'ध',
        36: 'क', 37: 'न', 38: 'प',
        39: 'फ', 40: 'ब', 41: 'भ',
        42: 'म', 43: 'य', 44: 'र',
        45: 'ल', 46: 'व', 47: 'ख',
        48: 'श', 49: 'ष',
        50: 'स', 51: 'ह', 52: 'क्ष',
        53: 'त्र', 54: 'ज्ञ', 55: 'ग',
        56: 'घ', 57: 'ङ', 58: 'च',
        59: 'छ', 60: 'ज', 61: 'झ'
    }
    
    return manual_mapping_dict.get(class_index, 'Unknown')

def predict_with_threshold(model, digit, threshold=0.55):
    # Reshape and normalize for prediction
    prediction = model.predict(digit.reshape(1, 28, 28, 1) / 255.)

    # Get the predicted class index and confidence
    predicted_class_index = np.argmax(prediction)
    confidence = prediction[0][predicted_class_index]

    # Check if confidence is above the threshold
    if confidence >= threshold:
        return manual_mapping(predicted_class_index)
    else:
        return 'Unknown'

def preprocess_image(image):
    # Apply median blur to reduce noise
    blur_image = cv2.medianBlur(image, 7)

    # Convert to grayscale
    grey = cv2.cvtColor(blur_image, cv2.COLOR_BGR2GRAY) if len(blur_image.shape) == 3 else blur_image

    # Apply adaptive thresholding
    thresh = cv2.adaptiveThreshold(grey, 200, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 41, 25)

    return thresh

def extract_preprocessed_digits(contours, image):
    preprocessed_digits = []

    for c in contours:
        # Additional preprocessing steps
        area = cv2.contourArea(c)
        x, y, w, h = cv2.boundingRect(c)
        aspect_ratio = float(w) / h

        # Contour area thresholding
        if area < 90:
            continue

        # Aspect ratio filtering
        if aspect_ratio < 0.2 or aspect_ratio > 2:
            continue

        # Bounding box size filtering
        if w < 10 or h < 10:
            continue

        # Extract the digit
        digit = image[y:y+h, x:x+w]

        # Resize the digit to match the input size used during training
        resized_digit = cv2.resize(digit, (28, 28))

        # Pad the digit
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)

        preprocessed_digits.append(padded_digit)

    return preprocessed_digits

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Save the uploaded image to a static folder
    with open(f"static/{file.filename}", "wb") as f:
        f.write(contents)

    # Load and preprocess the input image
    image_path = f"static/{file.filename}"
    original_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

    # Preprocess the image
    preprocessed_image = preprocess_image(original_image)

    # Find contours
    contours, hierarchy = cv2.findContours(preprocessed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sort contours based on x-coordinate
    boundingBoxes = [cv2.boundingRect(c) for c in contours]
    (contours, boundingBoxes) = zip(*sorted(zip(contours, boundingBoxes), key=lambda b: b[1][0], reverse=False))

    # Extract preprocessed digits
    preprocessed_digits = extract_preprocessed_digits(contours, preprocessed_image)

    alphabets_unseen = []
    for i, digit in enumerate(preprocessed_digits):
        # Resize digit to (28, 28)
        digit_resized = cv2.resize(digit, (28, 28))

        # Perform prediction with confidence thresholding
        pred = predict_with_threshold(model, digit_resized)
        alphabets_unseen.append(pred)

    # Return the prediction result
    return {"predicted_class": alphabets_unseen, "image_path": f"/static/{file.filename}"}