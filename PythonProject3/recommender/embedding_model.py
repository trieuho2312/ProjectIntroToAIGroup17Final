import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from .data_manager import ensure_dir

def build_embedding_model(df, out_dir="models"):
    tfidf = TfidfVectorizer(
        max_features=20000,
        min_df=3,
        max_df=0.7,
        stop_words='english',
        ngram_range=(1, 2)
    )
    tfidf_matrix = tfidf.fit_transform(df['cleaned_text'])

    nn_model = NearestNeighbors(metric='cosine', algorithm='brute')
    nn_model.fit(tfidf_matrix)

    ensure_dir(out_dir)
    joblib.dump(tfidf_matrix, os.path.join(out_dir, "tfidf_matrix.pkl"))
    joblib.dump(tfidf, os.path.join(out_dir, "tfidf_vectorizer.pkl"))
    joblib.dump(nn_model, os.path.join(out_dir, "nn_model.pkl"))

    print(" Saved TF-IDF, NearestNeighbors model, and vectorizer.")
