import streamlit as st
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# Load Model and Files
# -------------------------------
model = tf.keras.models.load_model("lsm_model.h5")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("max_len.pkl", "rb") as f:
    max_len = pickle.load(f)

# -------------------------------
# Predict Next Word Function
# -------------------------------
def predict_next_word(text):

    # Convert text to sequence
    token_list = tokenizer.texts_to_sequences([text])[0]

    # Pad sequence
    token_list = pad_sequences(
        [token_list],
        maxlen=max_len-1,
        padding='pre'
    )

    # Predict
    prediction = model.predict(token_list, verbose=0)

    # Get predicted index
    predicted_index = np.argmax(prediction, axis=-1)[0]

    # Convert index back to word
    output_word = ""

    for word, index in tokenizer.word_index.items():
        if index == predicted_index:
            output_word = word
            break

    return output_word

# -------------------------------
# Streamlit UI
# -------------------------------

st.set_page_config(
    page_title="Next Word Predictor",
    page_icon="📝",
    layout="centered"
)

st.title("📝 Next Word Prediction using LSTM")

st.write("Enter a word or sentence to predict the next word.")

user_input = st.text_input("Enter text")

if st.button("Predict"):

    if user_input.strip() == "":
        st.warning("Please enter some text.")

    else:

        predicted_word = predict_next_word(user_input)

        if predicted_word == "":
            st.error("No prediction found.")

        else:
            st.success(f"### Predicted Next Word: **{predicted_word}**")