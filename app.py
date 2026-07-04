# app.py
from flask import Flask, request, render_template_string
from transformers import pipeline

app = Flask(__name__)

# Load a sentiment analysis pipeline (downloads the model on first run)
sentiment_pipe = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Review Sentiment</title>
</head>
<body>
    <h1>Review Sentiment Analyzer</h1>
    <form method="post">
        <textarea name="review" rows="8" cols="60" placeholder="Write your review here...">{{ review_text }}</textarea><br><br>
        <input type="submit" value="Analyze Sentiment">
    </form>
    {% if result %}
        <h2>Result: {{ result }}</h2>
    {% endif %}
</body>
</html>
"""

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    review_text = ""
    if request.method == "POST":
        review_text = request.form.get("review", "")
        if review_text.strip():
            # Run sentiment analysis (truncation avoids max-length errors)
            output = sentiment_pipe(review_text, truncation=True)[0]
            # Map model labels to Positive/Negative
            label = output["label"]
            if label == "POSITIVE":
                result = "Positive"
            else:
                result = "Negative"
    return render_template_string(HTML_TEMPLATE, result=result, review_text=review_text)

if __name__ == "__main__":
    app.run(debug=True)