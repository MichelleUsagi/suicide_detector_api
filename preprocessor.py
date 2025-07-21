import joblib
from keras.preprocessing.sequence import pad_sequences

# Load the saved tokenizer
tokenizer = joblib.load("models/tokenizer.pkl")

def preprocess(text, tokenizer, max_length=200):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_length)
    return padded
