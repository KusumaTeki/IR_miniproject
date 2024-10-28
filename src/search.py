from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from preprocess import preprocess_text


def search(query, tfidf_matrix, vectorizer, articles, top_n=5):
    processed_query = preprocess_text(query)
    query_vec = vectorizer.transform([processed_query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    top_indices = np.argsort(cosine_similarities)[-top_n:][::-1]
    results = [(articles[i]['title'], cosine_similarities[i]) for i in top_indices]
    return results

# Example usage
if __name__ == "__main__":
    articles = load_articles()
    tfidf_matrix, vectorizer = create_tfidf_matrix(articles)
    query = "AI technology"
    search_results = search(query, tfidf_matrix, vectorizer, articles)
    for title, score in search_results:
        print(f"Title: {title}, Score: {score:.2f}")
