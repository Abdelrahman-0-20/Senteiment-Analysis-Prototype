import pandas as pd
from flask import Flask, request, render_template_string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os

app = Flask(__name__)

# Load data from CSV file
DATA_FILE = 'Movie_Review.csv'

def load_data():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found. Please place the reviews data in a CSV file with columns 'text' and 'sentiment'.")
    df = pd.read_csv(DATA_FILE, sep='\t', encoding='utf-8')  # assuming tab-separated
    return df['text'].tolist(), df['sentiment'].tolist()

# Train model on startup
try:
    texts, labels = load_data()
    vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    X = vectorizer.fit_transform(texts)
    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)
    print("Model trained successfully.")
except Exception as e:
    print(f"Error loading data or training model: {e}")
    model = None
    vectorizer = None

# HTML template (simple, no styles)
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Sentiment Analyzer</title>
</head>
<body>
    <h2>Enter your review:</h2>
    <form action="/predict" method="post">
        <textarea name="review" rows="6" cols="60">{{ review or '' }}</textarea><br><br>
        <input type="submit" value="Analyze">
    </form>
    {% if prediction %}
        <h3>Prediction: {{ prediction }}</h3>
    {% endif %}
    {% if error %}
        <p style="color:red;">{{ error }}</p>
    {% endif %}
</body>
</html>
"""

@app.route('/', methods=['GET'])
def index():
    return render_template_string(HTML)

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return render_template_string(HTML, error="Model not available. Please check data file.")

    review = request.form.get('review', '')
    if not review.strip():
        return render_template_string(HTML, review=review, error="Please enter some text.")

    # Transform and predict
    X_test = vectorizer.transform([review])
    pred = model.predict(X_test)[0]
    return render_template_string(HTML, review=review, prediction=pred)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)