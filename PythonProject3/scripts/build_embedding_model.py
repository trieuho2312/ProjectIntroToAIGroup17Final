import joblib
from recommender.embedding_model import build_embedding_model

if __name__ == "__main__":
    df = joblib.load("models/df_cleaned.pkl")
    build_embedding_model(df)