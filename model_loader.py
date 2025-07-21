from tensorflow.keras.models import load_model
import joblib
import os

def load_model_and_tokenizer():
    base_path = os.path.dirname(__file__)
    model_path = os.path.join(base_path, "models", "suicide_detector_model.h5")
    tokenizer_path = os.path.join(base_path, "models", "tokenizer.pkl")

    model = load_model(model_path)
    tokenizer = joblib.load(tokenizer_path)
    maxlen = 200  # Must match padding length used during training

    return model, tokenizer, maxlen
