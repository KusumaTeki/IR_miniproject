from flask import Flask, render_template, request
from preprocess import load_articles
from indexer import create_tfidf_matrix
from search import search

app = Flask(__name__)

# Load articles and create TF-IDF matrix at startup
articles = load_articles()
tfidf_matrix, vectorizer = create_tfidf_matrix(articles)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search_results():
    query = request.form['query']
    results = search(query, tfidf_matrix, vectorizer, articles)
    return render_template('results.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
