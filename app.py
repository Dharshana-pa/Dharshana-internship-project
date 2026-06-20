from flask import Flask, render_template, request
import PyPDF2
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/match', methods=['POST'])
def match():
    uploaded_file = request.files['resume']
    job_description = request.form['job_description']

    reader = PyPDF2.PdfReader(uploaded_file)

    text = ""

    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    cleaned_text = re.sub(r'[^a-zA-Z ]', ' ', text)
    cleaned_text = cleaned_text.lower()

    documents = [cleaned_text, job_description.lower()]

    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(documents)

    similarity = cosine_similarity(
        tfidf_matrix[0:1],
        tfidf_matrix[1:2]
    )

    percentage = round(similarity[0][0] * 100, 2)

    return f"<h1>Match Percentage: {percentage}%</h1>"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
