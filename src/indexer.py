from sklearn.feature_extraction.text import TfidfVectorizer

def create_tfidf_matrix(articles):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([article['processed_content'] for article in articles])
    return tfidf_matrix, vectorizer

# Example usage
if __name__ == "__main__":
    articles = load_articles()
    tfidf_matrix, vectorizer = create_tfidf_matrix(articles)
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
