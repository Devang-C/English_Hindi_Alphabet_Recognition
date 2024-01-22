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
model_path = 'models/resnet_alphabet_recognition_model.h5'  # Replace with the actual path to your saved model
model = load_model(model_path)

# Load the class indices mapping
class_indices_mapping = np.load('class_indices_mapping.npy', allow_pickle=True).item()

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


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    
    # Save the uploaded image to a static folder
    with open(f"static/{file.filename}", "wb") as f:
        f.write(contents)

    # Load and preprocess the input image
    input_image = cv2.imread(f"static/{file.filename}", cv2.IMREAD_GRAYSCALE)
    input_image = cv2.adaptiveThreshold(input_image,200,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,41,25)
    input_image = cv2.resize(input_image, (28, 28))
    input_image = input_image / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    # Perform prediction
    predictions = model.predict(input_image)

    # Decode the predictions using manual mapping
    predicted_class_index = np.argmax(predictions)
    predicted_class = manual_mapping(predicted_class_index)

    # Return the prediction result
    return {"predicted_class": predicted_class, "image_path": f"/static/{file.filename}"}
