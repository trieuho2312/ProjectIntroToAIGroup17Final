from recommender.data_manager import load_dataset, load_tfidf, load_model
import joblib
import logging
import numpy as np

class SongRecommender:
    def __init__(self):
        self.df = load_dataset()
        self.tfidf_matrix = load_tfidf()
        self.vectorizer = load_model("tfidf_vectorizer.pkl")
        self.nn_model = load_model("nn_model.pkl")

    def recommend(self, song_name, max_total=10, max_same_emotion=5):
        logging.info(f" Looking up song: {song_name}")
        idx_list = self.df[self.df['song'].str.lower() == song_name.lower()].index
        if len(idx_list) == 0:
            logging.warning(" Song not found in dataset.")
            return None, None
        idx = idx_list[0]
        original_emotion = self.df.iloc[idx]["emotion"]
        distances, indices = self.nn_model.kneighbors(
            self.tfidf_matrix[idx], n_neighbors=min(max_total + 1, len(self.df))
        )
        similar_indices = indices[0][1:]  # Bỏ bài hát chính nó
        results = self.df.iloc[similar_indices].copy()

        filtered = results[results["emotion"].str.lower() == original_emotion.lower()]
        others = results[results["emotion"].str.lower() != original_emotion.lower()]

        filtered = filtered.head(max_same_emotion).reset_index(drop=True)
        others = others.head(max_total - len(filtered)).reset_index(drop=True)
        return filtered, others

    def recommend_from_lyrics(self, lyrics, max_total=10, max_same_emotion=5):
        predicted_emotion = self._predict_emotion(lyrics)
        vec = self.vectorizer.transform([lyrics])
        distances, indices = self.nn_model.kneighbors(vec, n_neighbors=min(max_total, len(self.df)))
        top_indices = indices[0]
        top_results = self.df.iloc[top_indices].copy()

        similarity_scores = 1 - distances[0]
        emotion_matches = (top_results["emotion"].str.lower() == predicted_emotion.lower()).astype(int)
        final_scores = 0.7 * similarity_scores + 0.3 * emotion_matches

        top_results["cosine_similarity"] = similarity_scores
        top_results["emotion_match"] = emotion_matches
        top_results["final_score"] = final_scores

        filtered = top_results[top_results["emotion"].str.lower() == predicted_emotion.lower()]
        others = top_results[top_results["emotion"].str.lower() != predicted_emotion.lower()]

        filtered = filtered.sort_values(by="final_score", ascending=False).head(max_same_emotion).reset_index(drop=True)
        others = others.sort_values(by="final_score", ascending=False).head(max_total - len(filtered)).reset_index(drop=True)

        return filtered, others

    def _predict_emotion(self, text):
        from recommender.emotion_model import EmotionClassifier
        model = EmotionClassifier()
        return model.predict(text)
