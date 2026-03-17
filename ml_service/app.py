import os
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Load models and data from the same directory as this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
vectorizer = joblib.load(os.path.join(SCRIPT_DIR, "vectorizer.pkl"))
knn_model = joblib.load(os.path.join(SCRIPT_DIR, "knn_model.pkl"))
tfidf_matrix = joblib.load(os.path.join(SCRIPT_DIR, "tfidf.pkl"))
df = joblib.load(os.path.join(SCRIPT_DIR, "books_df.pkl"))

logger.info("Models loaded successfully. %d books available.", len(df))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "books": len(df)})


@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.json
        if not data or not data.get("prompt", "").strip():
            return jsonify({"error": "Please provide a non-empty prompt"}), 400

        prompt = data["prompt"].strip()
        logger.info("Recommendation request: %s", prompt[:100])

        prompt_vector = vectorizer.transform([prompt])
        distances, indices = knn_model.kneighbors(prompt_vector, n_neighbors=5)

        results = df.iloc[indices[0]][["book_title", "author", "genres", "book_details"]]
        return jsonify(results.to_dict(orient="records"))

    except Exception as e:
        logger.error("Error in /recommend: %s", e)
        return jsonify({"error": "Recommendation service encountered an error"}), 500


if __name__ == "__main__":
    app.run(port=5001)
