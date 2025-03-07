from flask import Flask, request, jsonify
import numpy as np
import json
#import openai
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load product embeddings
EMBEDDINGS_PATH = 'product_embeddings.json'
try:
    with open(EMBEDDINGS_PATH, 'r') as f:
        embeddings = json.load(f)
except FileNotFoundError:
    embeddings = {}


@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_id = data.get('user_id')
    purchased_products = data.get('orders', [])

    if not purchased_products:
        return jsonify({"recommendations": get_popular_products()})

    # Get vectors for purchased products
    product_vectors = [embeddings[str(pid)] for pid in purchased_products if str(pid) in embeddings]

    if not product_vectors:
        return jsonify({"recommendations": get_popular_products()})

    # Create weighted average (recent first)
    weights = np.linspace(0.1, 1.0, len(product_vectors))
    user_vector = np.average(product_vectors, axis=0, weights=weights)

    # Calculate similarities
    all_ids = list(embeddings.keys())
    all_vectors = np.array([embeddings[pid] for pid in all_ids])

    similarities = cosine_similarity([user_vector], all_vectors)[0]
    sorted_indices = np.argsort(similarities)[::-1]

    # Filter recommendations
    recommendations = []
    for idx in sorted_indices:
        candidate_id = int(all_ids[idx])
        if candidate_id not in purchased_products:
            recommendations.append(candidate_id)
            if len(recommendations) >= 5:
                break

    return jsonify({"recommendations": recommendations})


def get_popular_products():
    return [42, 56, 23, 87, 15]


if __name__ == '__main__':
    app.run(debug=True)
