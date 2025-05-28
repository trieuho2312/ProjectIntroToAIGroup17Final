import pandas as pd
import joblib
from recommender.preprocess import clean_text
from recommender.data_manager import ensure_dir

RAW_CSV = "data/spotify_dataset_10k.csv"
OUT_PKL = "models/df_cleaned.pkl"

if __name__ == "__main__":
    df = pd.read_csv(RAW_CSV)

    df = df.rename(columns={'Artist(s)': 'artist'})

    df = df[['artist', 'song', 'text', 'emotion']].dropna()
    df['cleaned_text'] = df['text'].apply(clean_text)

    ensure_dir("models")
    joblib.dump(df, OUT_PKL)
    print(f" Cleaned dataset saved to {OUT_PKL}")
