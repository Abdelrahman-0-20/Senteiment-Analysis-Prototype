import streamlit as st
from transformers import pipeline

@st.cache_resource
def load_model():
    return pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

pipe = load_model()

st.title("Review Sentiment Analyzer")

user_review = st.text_area(
    "Write your review here:",
    height=150,
    placeholder="Write your review here..."
)

if st.button("Analyze Sentiment"):
    if user_review.strip():
        output = pipe(user_review, truncation=True)[0]
        label = output["label"]
        result = "Positive" if label == "POSITIVE" else "Negative"
        st.subheader(f"Result: {result}")
    else:
        st.warning("Please enter a review.")
