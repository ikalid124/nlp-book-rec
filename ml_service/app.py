from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load models and data
vectorizer = joblib.load("vectorizer.pkl")
knn_model = joblib.load("knn_model.pkl")
tfidf_matrix = joblib.load("tfidf.pkl")
df = joblib.load("books_df.pkl")

@app.route("/recommend", methods=["POST"])
def recommend():
    try:
        data = request.json
        print("📥 Received data:", data)

        prompt = data.get("prompt", "")
        print("💬 Prompt:", prompt)

        prompt_vector = vectorizer.transform([prompt])
        distances, indices = knn_model.kneighbors(prompt_vector, n_neighbors=5)

        results = df.iloc[indices[0]][['book_title', 'author', 'genres','book_details']]
        print("✅ Recommendations ready")

        return jsonify(results.to_dict(orient="records"))

    except Exception as e:
        print("❌ Error in /recommend:", e)
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(port=5001, debug=True)
