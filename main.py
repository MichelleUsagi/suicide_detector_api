import logging
import csv
import os
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model_loader import load_model_and_tokenizer
from preprocessor import preprocess

# Load model, tokenizer, and maxlen
model, tokenizer, maxlen = load_model_and_tokenizer()

# Initialize FastAPI app
app = FastAPI()

# Setup structured logging
logging.basicConfig(
    filename="prediction_logs.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# CSV logging setup
csv_log_file = "prediction_logs.csv"
if not os.path.exists(csv_log_file):
    with open(csv_log_file, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["timestamp", "input_text", "prediction", "probability", "message"])

# For browser history
history = []

# Define request schema
class TextInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Suicide Ideation Detection API running."}

@app.post("/predict")
def predict(input_data: TextInput):
    if not input_data.text.strip():
        raise HTTPException(status_code=400, detail="Input text is empty.")

    try:
        # Preprocess and predict
        sequence = preprocess(input_data.text, tokenizer, maxlen)
        prediction = model.predict(sequence)[0][0]
        label = int(prediction >= 0.6)
        message = "High risk" if label == 1 else "Low risk"
        probability = float(prediction)
        timestamp = datetime.now().isoformat()

        # Create result
        result = {
            "timestamp": timestamp,
            "input_text": input_data.text,
            "prediction": label,
            "probability": probability,
            "message": message
        }

        # Save to history and CSV
        history.append(result)
        with open(csv_log_file, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow([timestamp, input_data.text, label, probability, message])

        # Log to system
        logging.info(f"Prediction: {result}")
        return result

    except Exception as e:
        logging.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed.")

@app.get("/history")
def get_history():
    return history[-20:]
