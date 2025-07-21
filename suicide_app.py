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
# Ensure this path is correct relative to your app's root
model = tf.keras.models.load_model("models/suicide_detector_model.h5") 

# Load tokenizer
with open("models/tokenizer.json", "r") as f:
    tokenizer_json = json.load(f)
    tokenizer = tokenizer_from_json(tokenizer_json)

# Correct max_len to match your model's input shape (e.g., 200)
def preprocess_text(text, max_len=200): # <--- Make sure this is 200
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

                # Update chat_history in session_state
                st.session_state.chat_history.append((user_input, reply))

                # Log prediction to CSV
                with open(CSV_LOG_FILE, "a", encoding="utf-8") as f:
                    timestamp = datetime.now().isoformat()
                    f.write(f"{timestamp},{user_input},{prediction},{prob:.2f}\n")

            except Exception as e:
                st.session_state.chat_history.append((user_input, f"Error: {e}"))
            
            # --- IMPORTANT: Do NOT display history here ---
            # Remove the loop that iterates through st.session_state.chat_history here.

# Tab 2 - History
with menu[1]:
    st.subheader("Prediction History")
    
    # Display the current session's chat history (optional, if you want it distinct from CSV)
    if st.session_state.chat_history:
        st.markdown("#### Current Session Log:")
        for user_msg, bot_msg in reversed(st.session_state.chat_history):
            st.markdown(f"**You:** {user_msg}")
            st.markdown(f"<div style='margin-left:20px;color:#2c3e50;'>💡 <i>{bot_msg}</i></div>", unsafe_allow_html=True)
        st.markdown("---") # Separator

    # Display history loaded from CSV file
    st.markdown("#### Full History from File:")
    if os.path.exists(CSV_LOG_FILE):
        try:
            # Assuming CSV_LOG_FILE has no header initially, or adjust names as needed
            history_df = pd.read_csv(CSV_LOG_FILE, header=None, names=["Timestamp", "Input", "Prediction", "Confidence"])
            st.dataframe(history_df)
        except pd.errors.EmptyDataError:
            st.info("No prediction history yet.")
        except Exception as e:
            st.error(f"Error loading history: {e}")
    else:
        st.info("No prediction history file found yet.")

# Tab 3 - Feedback
with menu[2]:
    st.subheader("Provide Feedback")
    feedback_text = st.text_area("Your Feedback:", "", key="feedback_input") # Added a key
    if st.button("Submit Feedback", key="submit_feedback_button"): # Added a key
        if feedback_text.strip():
            with open(FEEDBACK_FILE, "a", encoding="utf-8") as f:
                timestamp = datetime.now().isoformat()
                f.write(f"{timestamp},{feedback_text}\n")
            st.success("Thank you for your feedback!")
            st.session_state.feedback_input = "" # Clear the text area using session_state key
        else:
            st.warning("Please enter some feedback.")