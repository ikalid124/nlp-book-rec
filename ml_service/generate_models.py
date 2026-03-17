"""
Generate .pkl model files from the Kaggle books dataset.

Downloads the dataset, preprocesses it, fits a TF-IDF vectorizer and
NearestNeighbors model, and saves 4 .pkl files for the Flask app.

Requires KAGGLE_USERNAME and KAGGLE_KEY environment variables.
"""

import os
import subprocess
import zipfile
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET = "dk123891/books-dataset-goodreadsmay-2024"
ZIP_FILE = os.path.join(SCRIPT_DIR, "books-dataset-goodreadsmay-2024.zip")
CSV_FILE = os.path.join(SCRIPT_DIR, "Book_Details.csv")


def download_dataset():
    """Download and extract the Kaggle dataset."""
    print("Downloading dataset from Kaggle...")
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", DATASET, "-p", SCRIPT_DIR],
        check=True,
    )
    print("Extracting dataset...")
    with zipfile.ZipFile(ZIP_FILE, "r") as zf:
        zf.extract("Book_Details.csv", SCRIPT_DIR)


def preprocess(df):
    """Select columns, drop nulls, build combined text feature."""
    df = df[["book_title", "book_details", "author", "genres", "publication_info"]].copy()
    df.dropna(subset=["book_details"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df["book_details"] = df["book_details"].fillna("")
    df["genres"] = df["genres"].fillna("")
    df["publication_info"] = df["publication_info"].fillna("")

    df["combined_text"] = (
        df["book_title"] + " " + df["publication_info"] + " " +
        df["genres"] + " " + df["book_details"]
    ).str.lower().str.strip()

    return df


def build_models(df):
    """Fit TF-IDF vectorizer and NearestNeighbors on all data."""
    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df["combined_text"])

    print("Fitting NearestNeighbors model...")
    knn_model = NearestNeighbors(n_neighbors=5, metric="cosine")
    knn_model.fit(tfidf_matrix)

    return vectorizer, tfidf_matrix, knn_model


def save_artifacts(vectorizer, tfidf_matrix, knn_model, df):
    """Save all model artifacts as .pkl files."""
    paths = {
        "vectorizer.pkl": vectorizer,
        "knn_model.pkl": knn_model,
        "tfidf.pkl": tfidf_matrix,
        "books_df.pkl": df[["book_title", "author", "genres", "book_details"]],
    }
    for name, obj in paths.items():
        path = os.path.join(SCRIPT_DIR, name)
        joblib.dump(obj, path)
        print(f"Saved {path}")


def cleanup():
    """Remove downloaded dataset files."""
    for path in [ZIP_FILE, CSV_FILE]:
        if os.path.exists(path):
            os.remove(path)
            print(f"Removed {path}")
    # Also clean up extra files from the zip
    for extra in ["book_reviews.db", "books.db"]:
        p = os.path.join(SCRIPT_DIR, extra)
        if os.path.exists(p):
            os.remove(p)
            print(f"Removed {p}")


def main():
    download_dataset()

    print("Loading CSV...")
    df = pd.read_csv(CSV_FILE)

    df = preprocess(df)
    print(f"Dataset: {len(df)} books after preprocessing")

    vectorizer, tfidf_matrix, knn_model = build_models(df)
    save_artifacts(vectorizer, tfidf_matrix, knn_model, df)

    cleanup()
    print("Done! Model files generated successfully.")


if __name__ == "__main__":
    main()
