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
        20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z', 26: 'character_10_yna',
        27: 'character_11_taamatar', 28: 'character_12_thaa', 29: 'character_13_daa',
        30: 'character_14_dhaa', 31: 'character_15_adna', 32: 'character_16_tabala',
        33: 'character_17_tha', 34: 'character_18_da', 35: 'character_19_dha',
        36: 'character_1_ka', 37: 'character_20_na', 38: 'character_21_pa',
        39: 'character_22_pha', 40: 'character_23_ba', 41: 'character_24_bha',
        42: 'character_25_ma', 43: 'character_26_yaw', 44: 'character_27_ra',
        45: 'character_28_la', 46: 'character_29_waw', 47: 'character_2_kha',
        48: 'character_30_motosaw', 49: 'character_31_petchiryakha',
        50: 'character_32_patalosaw', 51: 'character_33_ha', 52: 'character_34_chhya',
        53: 'character_35_tra', 54: 'character_36_gya', 55: 'character_3_ga',
        56: 'character_4_gha', 57: 'character_5_kna', 58: 'character_6_cha',
        59: 'character_7_chha', 60: 'character_8_ja', 61: 'character_9_jha'
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