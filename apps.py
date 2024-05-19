import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import numpy as np

@st.cache(allow_output_mutation=True)
def load_resources():
    # Load tokenizer
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Load model
    model = load_model('model_berita_classification.h5')

    # Load maxlen
    with open('maxlen.txt', 'r') as f:
        maxlen = int(f.read())

    return tokenizer, model, maxlen

# Load resources
tokenizer, model, maxlen = load_resources()

pad_type = 'pre'
trunc_type = 'pre'
categories = ['Kesehatan', 'Keuangan', 'Kuliner', 'Olahraga', 'Otomotif', 'Pariwisata', 'Pendidikan']

st.title('News Category Prediction')
st.write('Input a news headline or short text to predict its category.')

# Input text
input_text = st.text_area("Enter text here:")

if st.button('Predict'):
    if input_text:
        # Preprocess the input text
        sekuens_data_baru = tokenizer.texts_to_sequences([input_text])
        padded_data_baru = pad_sequences(sekuens_data_baru, padding=pad_type, truncating=trunc_type, maxlen=maxlen)

        # Make prediction
        hasil_prediksi = model.predict(padded_data_baru)

        # Get the category with the highest probability
        indeks_tertinggi = np.argmax(hasil_prediksi)
        kategori_prediksi = categories[indeks_tertinggi]

        st.write(f"Predicted Category: **{kategori_prediksi}**")
        st.write(f"Prediction Confidence: {hasil_prediksi[0][indeks_tertinggi]:.2f}")
    else:
        st.write("Please enter some text for prediction.")
