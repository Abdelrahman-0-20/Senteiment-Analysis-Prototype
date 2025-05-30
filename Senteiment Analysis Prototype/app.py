import pandas as pd
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# Load the model and scaler
model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))

# User input for movie review
review = st.text_input('Enter Movie Review')

if st.button('Predict'):
    # Transform the input and make prediction
    review_scale = scaler.transform([review]).toarray()
    result = model.predict(review_scale)
    
    # Display the result
    if result[0] == 0:
        st.error("Negative Review ")
    else:
        st.success("Positive Review ")
