import streamlit as st
from PIL import Image
import os
from datetime import datetime
import pandas as pd

import tensorflow as tf
import json
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load the trained model
model = tf.keras.models.load_model("models/suicide_detector_model.h5")

# Load tokenizer
with open("models/tokenizer.json", "r") as f:
    tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)

def preprocess_text(text, max_len=100):
    sequence = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(sequence, maxlen=max_len)
    return padded

LOGO_PATH = "logo.png"
CSV_LOG_FILE = "prediction_logs.csv"
FEEDBACK_FILE = "feedback_log.csv"

# Page configuration
st.set_page_config(page_title="MindMate – Emotional Support App", page_icon="🧠", layout="wide")

# Load and center the logo
if os.path.exists(LOGO_PATH):
    st.image(LOGO_PATH, width=120)

# Title and description
st.markdown("""
    <h1 style='text-align: center; color: #2c3e50;'>MindMate</h1>
    <h4 style='text-align: center; color: #34495e;'>Your supportive companion for emotional check-ins</h4>
""", unsafe_allow_html=True)

# Tabs for chatbot, history, and feedback
menu = st.tabs(["💬 Chatbot", "📜 History", "📝 Feedback"])

# Tab 1 - Chatbot
with menu[0]:
    st.subheader("Chat with MindMate")
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_input = st.text_input("You:", "", key="chat_input")

    if st.button("Send"):
        if user_input.strip():
            try:
                input_seq = preprocess_text(user_input)
                prob = model.predict(input_seq)[0][0]
                prediction = 1 if prob > 0.5 else 0
                label = "🚨 High Risk" if prediction == 1 else "✅ Low Risk"
                reply = f"Prediction: {label}\nConfidence: {prob:.2f}"

                # Update chat
                st.session_state.chat_history.append((user_input, reply))

                # Log prediction
                with open(CSV_LOG_FILE, "a", encoding="utf-8") as f:
                    timestamp = datetime.now().isoformat()
                    f.write(f"{timestamp},{user_input},{prediction},{prob:.2f}\n")

            except Exception as e:
                st.session_state.chat_history.append((user_input, f"Error: {e}"))

    for user_msg, bot_msg in reversed(st.session_state.chat_history):
        st.markdown(f"**You:** {user_msg}")
        st.markdown(f"<div style='margin-left:20px;color:#2c3e50;'>💡 <i>{bot_msg}</i></div>", unsafe_allow_html=True)


# Tab 2 - History
with menu[1]:
    st.subheader("Prediction History")
    if os.path.exists(CSV_LOG_FILE): # <-- This is the corrected line
        try:
            history_df = pd.read_csv(CSV_LOG_FILE, header=None, names=["Timestamp", "Input", "Prediction", "Confidence"])
            st.dataframe(history_df)
        except pd.errors.EmptyDataError:
            st.info("No prediction history yet.")
    else:
        st.info("No prediction history yet.")

# Tab 3 - Feedback
with menu[2]:
    st.subheader("Provide Feedback")
    feedback_text = st.text_area("Your Feedback:", "")
    if st.button("Submit Feedback"):
        if feedback_text.strip():
            with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
                timestamp = datetime.now().isoformat()
                f.write(f"{timestamp},{feedback_text}\n")
            st.success("Thank you for your feedback!")
            feedback_text = "" # Clear the text area
        else:
            st.warning("Please enter some feedback.")