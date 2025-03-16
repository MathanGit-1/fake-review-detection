import tensorflow as tf
import tensorflow_hub as hub
from data_preprocessing import preprocess_text

# Load the trained model
model = tf.keras.models.load_model('fake_review_model_tf.h5')

# Load the Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

def predict_review(review):
    """
    Predicts whether a review is fake or real.
    """
    processed_review = preprocess_text(review)  # Preprocess the review
    review_embedding = embed([processed_review]).numpy()  # Convert to embeddings
    prediction = model.predict(review_embedding)[0][0]  # Get prediction

    return "Fake" if prediction > 0.5 else "Real"

# Sample reviews for testing
test_reviews = [
    "This product is amazing! I would highly recommend it to everyone.",
    "Worst experience ever. Total waste of money.",
    "This is the best phone I've ever used. The camera is stunning!",
    "I received this product for free in exchange for a review, and it's really great!",
    "I highly doubt this review is genuine. Seems very generic.",
]

# Run predictions
for review in test_reviews:
    print(f"Review: {review}")
    print(f"Prediction: {predict_review(review)}\n")
