import pickle as pk
import streamlit as st

# ====== Clean Styling ======
st.markdown("""
<style>
    /* Remove top padding and header */
    .stApp {
        padding-top: 0rem;
    }
    header {
        display: none;
    }
    
    /* Main container */
    .main-container {
        max-width: 700px;
        margin: 0 auto;
        padding: 2rem 1.5rem;
    }
    
    /* Title */
    h1 {
        color: #2c3e50;
        font-size: 2rem;
        margin-bottom: 2rem;
        font-weight: 600;
    }
    
    /* Input field */
    .stTextInput>div>div>input {
        padding: 12px;
        font-size: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 6px;
    }
    
    /* Button */
    .stButton>button {
        width: 100%;
        padding: 12px;
        background-color: #4a6bdf;
        color: white;
        font-size: 1rem;
        border: none;
        border-radius: 6px;
        margin-top: 1rem;
    }
    
    /* Results */
    .stAlert {
        padding: 1rem;
        border-radius: 6px;
        margin-top: 1.5rem;
    }
    
    /* Footer */
    .footer {
        margin-top: 3rem;
        color: #7f8c8d;
        font-size: 0.9rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ====== App Layout ======
st.markdown('<div class="main-container">', unsafe_allow_html=True)

st.title("🎬 Movie Review Analysis")

# Load model and scaler
try:
    model = pk.load(open('model.pkl', 'rb'))
    scaler = pk.load(open('scaler.pkl', 'rb'))
except FileNotFoundError:
    st.error("🔍 Model files not found")
    st.stop()

# User input
review = st.text_input('📝 Enter your movie review:', 
                      placeholder="The acting was superb but the plot was confusing...")

if st.button('🔎 Analyze'):
    if not review.strip():
        st.warning("⚠️ Please enter a review")
    else:
        try:
            # Transform and predict
            review_scale = scaler.transform([review]).toarray()
            result = model.predict(review_scale)[0]
            
            # Display result
            if result == 0:
                st.error("Negative Review")
            else:
                st.success("Positive Review")
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

# Footer
st.markdown("""
<div class="footer">
    Sentiment analysis powered by machine learning
</div>
</div>
""", unsafe_allow_html=True)