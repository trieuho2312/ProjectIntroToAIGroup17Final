import os
import joblib
import pandas as pd

MODELS_DIR = "models"

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_model(obj, filename):
    ensure_dir(MODELS_DIR)
    joblib.dump(obj, os.path.join(MODELS_DIR, filename))

def load_model(filename):
    return joblib.load(os.path.join(MODELS_DIR, filename))

def load_dataset():
    return joblib.load(os.path.join(MODELS_DIR, "df_cleaned.pkl"))

def load_nn_model():
    return joblib.load(os.path.join(MODELS_DIR, "nn_model.pkl"))

def load_tfidf():
    return joblib.load(os.path.join(MODELS_DIR, "tfidf_matrix.pkl"))